import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

import anthropic


class CoachChatInterface:
    """Interactive chat interface for soccer coaching insights."""

    def __init__(self, analysis_json_path: str):
        """
        Initialize chat interface with match data.

        Args:
            analysis_json_path: Path to LLM-formatted analysis JSON
        """
        # Load API key
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            print("❌ ERROR: ANTHROPIC_API_KEY not found")
            print("\nTo fix this:")
            print("  1. Add to .env file in project root:")
            print("     ANTHROPIC_API_KEY=sk-ant-...")
            print("  2. Or set environment variable:")
            print("     export ANTHROPIC_API_KEY='sk-ant-...'")
            print("\nGet your API key at: https://console.anthropic.com/")
            sys.exit(1)

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"

        # Load match data
        self.json_path = analysis_json_path
        with open(analysis_json_path, 'r') as f:
            self.match_data = json.load(f)

        # Conversation history
        self.conversation_history = []

        # Assess data quality first
        self.data_quality = self._assess_data_quality()

        # Build system context once
        self.system_context = self._build_system_context()

        print(f"✓ Loaded match data from: {Path(analysis_json_path).name}")

        # Warn about data quality if needed
        if self.data_quality['severity'] == 'critical':
            print(f"\n⚠️  WARNING: Data quality is limited")
            print(f"   - {self.data_quality['warning']}")
            print(f"   - Insights will be general/limited\n")

    def _assess_data_quality(self) -> Dict:
        """
        Assess quality of input data to adjust LLM behavior.

        Returns:
            Dict with quality assessment and warnings
        """
        data = self.match_data

        # Check for critical missing data
        team_stats = data.get('team_statistics', {})
        players = data.get('player_statistics', {})

        team_0 = team_stats.get('team_0', {})
        team_1 = team_stats.get('team_1', {})

        passes_0 = team_0.get('passing', {}).get('passes_completed', 0)
        passes_1 = team_1.get('passing', {}).get('passes_completed', 0)
        possession_0 = team_0.get('possession', {}).get('possession_percentage', 0)
        possession_1 = team_1.get('possession', {}).get('possession_percentage', 0)

        # Calculate issues
        issues = []
        severity = 'good'

        # Critical: No passes detected
        if passes_0 == 0 and passes_1 == 0:
            issues.append("No passes detected - passing analysis unavailable")
            severity = 'critical'

        # Critical: No possession data
        if possession_0 == 0 and possession_1 == 0:
            issues.append("No possession data - tactical analysis limited")
            if severity != 'critical':
                severity = 'critical'

        # Warning: Short clip
        duration = data.get('match_summary', {}).get('duration_seconds', 0)
        if duration < 60:
            issues.append(f"Very short clip ({duration}s) - patterns may not be meaningful")
            if severity == 'good':
                severity = 'warning'

        # Check for zero touches
        total_touches = sum(
            p.get('possession', {}).get('touches', 0)
            for p in players.values()
        )
        if total_touches == 0:
            issues.append("No ball touches recorded - possession analysis unavailable")
            severity = 'critical'

        return {
            'severity': severity,
            'issues': issues,
            'warning': issues[0] if issues else None,
            'has_passes': passes_0 > 0 or passes_1 > 0,
            'has_possession': possession_0 > 0 or possession_1 > 0,
            'has_touches': total_touches > 0,
            'duration_seconds': duration
        }

    def _build_system_context(self) -> str:
        """
        Build system context from match data with honest quality assessment.
        """
        data = self.match_data
        quality = self.data_quality

        # Extract key info
        summary = data.get('match_summary', {})
        team_stats = data.get('team_statistics', {})
        players = data.get('player_statistics', {})

        duration = summary.get('duration_seconds', 0)

        # Team summaries
        team_0 = team_stats.get('team_0', {})
        team_1 = team_stats.get('team_1', {})

        team_0_overall = team_0.get('overall', {})
        team_1_overall = team_1.get('overall', {})

        # Top performers
        top_players = self._get_top_performers(players, n=5)

        # Build data quality warning
        quality_warning = ""
        if quality['severity'] == 'critical':
            quality_warning = f"""
CRITICAL DATA LIMITATIONS:
This analysis has SEVERE data quality issues. Your responses MUST acknowledge these limitations:
- {' / '.join(quality['issues'])}

You can ONLY analyze:
- Player movement and distance (this IS tracked)
- Relative speed comparisons between players
- Team movement patterns (who covered more ground)

You CANNOT accurately analyze:
- Passing patterns (no pass data)
- Possession tactics (no possession data)
- Tactical formations (insufficient data)
- Ball control or touches (not reliably tracked)

RESPONSE STYLE:
- Start with honest caveat about data limitations
- Focus ONLY on what IS tracked (movement, speed, distance)
- Use plain conversational language - NO markdown, NO bullet points, NO asterisks
- Be helpful but honest - don't make up insights from missing data
"""
        else:
            quality_warning = """
DATA QUALITY:
Early-stage computer vision with ~80% accuracy. Focus on trends and patterns, not precision.
"""

        context = f"""You are a helpful soccer coach analyzing match data from an AI vision system.

{quality_warning}

MATCH CONTEXT:
Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)
Teams: {team_0.get('name', 'Team 0')} (Blue) vs {team_1.get('name', 'Team 1')} (Red)
Players Tracked: {len(players)} total

TEAM 0 (Blue):
Total Distance: {team_0_overall.get('total_distance_km', 0):.2f} km
Avg Speed: {team_0_overall.get('average_speed_kmh', 0):.1f} km/h
Passes Detected: {team_0.get('passing', {}).get('passes_completed', 0)}
Possession: {team_0.get('possession', {}).get('possession_percentage', 0):.0f}%

TEAM 1 (Red):
Total Distance: {team_1_overall.get('total_distance_km', 0):.2f} km
Avg Speed: {team_1_overall.get('average_speed_kmh', 0):.1f} km/h
Passes Detected: {team_1.get('passing', {}).get('passes_completed', 0)}
Possession: {team_1.get('possession', {}).get('possession_percentage', 0):.0f}%

TOP 5 MOST ACTIVE PLAYERS:
{top_players}

RESPONSE FORMAT - CRITICAL:
- Write in plain, natural conversation style
- NO markdown formatting (no **, no bullet points, no headers, no lists)
- NO asterisks or special characters for emphasis
- Just talk like a human coach would
- Use line breaks for readability, but keep it conversational
- Be concise - 3-5 sentences unless asked for detail

RESPONSE APPROACH:
1. If question requires missing data (passes/possession), say so honestly upfront
2. Focus on what IS available (movement, speed, distance comparisons)
3. Give practical coaching insights based on available data only
4. Don't speculate beyond what the data shows
5. Keep it conversational and helpful despite limitations"""

        return context

    def _get_top_performers(self, players: Dict, n: int = 5) -> str:
        """Format top N performers by movement/activity."""

        # Score by distance and speed (what we DO have)
        scored = []
        for pid, stats in players.items():
            performance = stats.get('performance', {})

            score = (
                performance.get('distance_covered_km', 0) * 100 +
                performance.get('average_speed_kmh', 0) * 2
            )

            scored.append((pid, stats, score))

        scored.sort(key=lambda x: x[2], reverse=True)

        lines = []
        for rank, (pid, stats, _) in enumerate(scored[:n], 1):
            team = stats.get('team', '?')
            pos = stats.get('position_estimate', 'Unknown')
            perf = stats.get('performance', {})

            lines.append(
                f"{rank}. Player #{pid} (Team {team}, {pos}): "
                f"{perf.get('distance_covered_km', 0):.2f}km, "
                f"{perf.get('average_speed_kmh', 0):.1f} km/h avg"
            )

        return '\n'.join(lines)

    def ask_question(self, question: str) -> str:
        """
        Ask Claude a question about the match.

        Args:
            question: Coach's question

        Returns:
            Claude's response
        """
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": question
        })

        # Call Claude API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,  # Shorter responses
                temperature=0.3,  # More factual
                system=self.system_context,
                messages=self.conversation_history
            )

            answer = response.content[0].text

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })

            return answer

        except anthropic.APIError as e:
            return f"❌ API Error: {e}"
        except Exception as e:
            return f"❌ Error: {e}"

    def run_chat_loop(self):
        """Run interactive chat loop."""

        print("\n" + "="*70)
        print("⚽ SOCCER COACH AI - Interactive Analysis")
        print("="*70)
        print("\nMatch loaded and ready for questions!")
        print("\nCommands:")
        print("  - Ask any question about the match")
        print("  - Type 'quit' or 'exit' to end")
        print("  - Type 'summary' for match overview")
        print("  - Type 'help' for example questions")

        if self.data_quality['severity'] == 'critical':
            print("\n⚠️  Note: Limited data - focus on movement/speed questions")

        print("\n" + "="*70 + "\n")

        while True:
            try:
                # Get user input
                question = input("🎤 Coach: ").strip()

                if not question:
                    continue

                # Handle commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for using Soccer Coach AI!")
                    break

                if question.lower() == 'help':
                    self._show_example_questions()
                    continue

                if question.lower() == 'summary':
                    if self.data_quality['severity'] == 'critical':
                        question = "Give me a brief summary focusing on what data IS available - player movement and work rate."
                    else:
                        question = "Give me a 3-sentence summary of this match."

                # Ask Claude
                print("\n🤖 AI Coach: ", end='', flush=True)
                answer = self.ask_question(question)
                print(answer)
                print()

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break

    def _show_example_questions(self):
        """Show example questions based on data quality."""

        if self.data_quality['severity'] == 'critical':
            # Limited data - show movement-focused questions
            examples = [
                "Which team was more active in terms of movement?",
                "Who covered the most distance?",
                "Which players showed the highest work rate?",
                "How did the teams compare in average speed?",
                "Who was the fastest player?",
            ]
            print("\n💡 Questions you CAN ask (based on available data):")
        else:
            # Full data - show all questions
            examples = [
                "Which team dominated and why?",
                "Who were the most active players?",
                "What should Team 0 work on in training?",
                "Which players covered the most distance?",
                "Who was the fastest player?",
                "What tactical adjustments would you recommend?",
            ]
            print("\n💡 Example Questions:")

        for i, q in enumerate(examples, 1):
            print(f"  {i}. {q}")
        print()


def main():
    """Main entry point."""

    if len(sys.argv) < 2:
        print("Usage: python coach_chat.py <path_to_analysis_json>")
        print("\nExample:")
        print("  python coach_chat.py output_videos/match_analysis_data_llm.json")
        sys.exit(1)

    json_path = sys.argv[1]

    # Check file exists
    if not Path(json_path).exists():
        print(f"❌ File not found: {json_path}")
        sys.exit(1)

    # Initialize and run chat
    chat = CoachChatInterface(json_path)
    chat.run_chat_loop()


if __name__ == "__main__":
    main()