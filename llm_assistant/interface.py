#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import anthropic


class CoachChatInterface:
    """Interactive chat interface for soccer coaching insights."""

    def __init__(self, analysis_json_path: str):
        """
        Initialize chat interface with match data.

        Args:
            analysis_json_path: Path to LLM-formatted analysis JSON
        """
        # Load API key (from .env file or environment variable)
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

        # Build system context once
        self.system_context = self._build_system_context()

        print(f"✓ Loaded match data from: {Path(analysis_json_path).name}")

    def _build_system_context(self) -> str:
        """
        Build system context from match data.

        This creates a concise summary that Claude can use for all questions.
        """
        data = self.match_data

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

        context = f"""You are an expert soccer coach analyzing match data from an AI vision system.

**MATCH CONTEXT:**
- Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)
- Teams: {team_0.get('name', 'Team 0')} vs {team_1.get('name', 'Team 1')}
- Players Tracked: {len(players)} total

**TEAM 0 ({team_0.get('name', 'Team 0')}) - Blue:**
- Total Distance: {team_0_overall.get('total_distance_km', 0):.2f} km
- Avg Speed: {team_0_overall.get('average_speed_kmh', 0):.1f} km/h
- Passes: {team_0.get('passing', {}).get('passes_completed', 0)}
- Possession: {team_0.get('possession', {}).get('possession_percentage', 0):.0f}%

**TEAM 1 ({team_1.get('name', 'Team 1')}) - Red:**
- Total Distance: {team_1_overall.get('total_distance_km', 0):.2f} km
- Avg Speed: {team_1_overall.get('average_speed_kmh', 0):.1f} km/h
- Passes: {team_1.get('passing', {}).get('passes_completed', 0)}
- Possession: {team_1.get('possession', {}).get('possession_percentage', 0):.0f}%

**TOP 5 PERFORMERS:**
{top_players}

**DATA QUALITY NOTE:**
This is early-stage computer vision with ~80% accuracy. Focus on:
- Comparative patterns (Team A vs Team B)
- Relative player performance (Player X vs Player Y)
- Statistical trends (not individual frame precision)

**YOUR ROLE:**
Answer the coach's questions with:
1. Direct, actionable insights
2. Specific references to the data
3. Honest caveats when data is limited
4. Coaching recommendations when appropriate

Keep responses concise and focused on what the coach asked."""

        return context

    def _get_top_performers(self, players: Dict, n: int = 5) -> str:
        """Format top N performers."""

        # Score each player
        scored = []
        for pid, stats in players.items():
            performance = stats.get('performance', {})
            passing = stats.get('passing', {})
            possession = stats.get('possession', {})

            score = (
                performance.get('distance_covered_km', 0) * 100 +
                passing.get('passes_completed', 0) * 10 +
                possession.get('touches', 0) * 5
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
                max_tokens=2048,
                temperature=0.3,  # Lower = more factual
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
        print("  - Type 'summary' for a quick match overview")
        print("  - Type 'help' for example questions")
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
                    question = "Give me a 3-sentence summary of this match highlighting the most important insights."

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
        """Show example questions to ask."""
        examples = [
            "Which team dominated and why?",
            "Who were the most active players?",
            "What should Team 0 work on in training?",
            "How did the teams compare in terms of pressing?",
            "Which players covered the most distance?",
            "What tactical adjustments would you recommend?",
            "Who was the most efficient passer?",
            "How did Team 1's formation affect their performance?",
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