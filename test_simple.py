import requests
import json
import re


def strip_ansi_codes(text):
    """Remove ANSI color codes from text"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def analyze_raw_output():
    """Analyze raw output from different endpoints"""

    print("=" * 80)
    print("ANALYZING RAW OUTPUT")
    print("=" * 80)

    test_input = "A 60-year-old woman with a history of rheumatoid arthritis and chronic kidney disease presents with increasing fatigue, shortness of breath, and lightheadedness. She reports no chest pain or bleeding. Recent labs reveal hemoglobin at 7.0 g/dL. Her medications include methotrexate and lisinopril. Physical exam is notable for pallor and mild ankle swelling. She requires evaluation for anemia of chronic disease and possible medication effects."

    # Test 1: Get raw crew output
    print("\n1. TESTING RAW ENDPOINT (/api/v1/analyze-raw):")
    print("-" * 40)

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/analyze-raw",
            json={"patient_input": test_input}
        )

        if response.status_code == 200:
            result = response.json()
            raw_output = result.get('raw_crew_output', '')

            print(f"✓ Raw output length: {len(raw_output)} chars")
            print(f"✓ Processing time: {result.get('processing_time_ms')} ms")

            # Save raw output
            with open("raw_crew_output.txt", "w", encoding="utf-8") as f:
                f.write(raw_output)
            print("✓ Saved raw output to: raw_crew_output.txt")

            # Save cleaned output
            cleaned = strip_ansi_codes(raw_output)
            with open("cleaned_crew_output.txt", "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"✓ Saved cleaned output to: cleaned_crew_output.txt")
            print(f"✓ Size reduction after cleaning: {len(raw_output) - len(cleaned)} chars")

            # Show first 500 chars of each
            print("\nRAW OUTPUT (first 500 chars):")
            print(raw_output[:500])

            print("\nCLEANED OUTPUT (first 500 chars):")
            print(cleaned[:500])

    except Exception as e:
        print(f"Error testing raw endpoint: {e}")

    # Test 2: Debug execution endpoint
    print("\n2. TESTING DEBUG ENDPOINT (/api/v1/debug-execution):")
    print("-" * 40)

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/debug-execution",
            json={"patient_input": test_input}
        )

        if response.status_code == 200:
            result = response.json()

            print(f"✓ Stdout captured: {result.get('stdout_length', 0)} chars")
            print(f"✓ Result type: {result.get('final_result_type')}")

            stdout = result.get('captured_stdout', '')
            if stdout:
                with open("debug_stdout.txt", "w", encoding="utf-8") as f:
                    f.write(stdout)
                print("✓ Saved stdout to: debug_stdout.txt")

                # Check what's in stdout
                print("\nContent analysis:")
                if "Medical NLP Specialist" in stdout:
                    print("  ✓ Contains Medical NLP Specialist output")
                if "\x1b[" in stdout or "\033[" in stdout:
                    print("  ⚠️ Contains ANSI escape codes")
                if "Final Answer:" in stdout:
                    print("  ✓ Contains Final Answer sections")

    except Exception as e:
        print(f"Error testing debug endpoint: {e}")

    # Test 3: Analyze main endpoint agent outputs
    print("\n3. ANALYZING MAIN ENDPOINT AGENT OUTPUTS:")
    print("-" * 40)

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/analyze",
            json={"patient_input": test_input}
        )

        if response.status_code == 200:
            result = response.json()
            agent_outputs = result.get('data', {}).get('agent_analysis', {})

            for agent, output in agent_outputs.items():
                if output:
                    has_ansi = bool(re.search(r'\x1B|\033', output))
                    cleaned = strip_ansi_codes(output)
                    reduction = len(output) - len(cleaned)

                    print(f"\n{agent}:")
                    print(f"  Original: {len(output)} chars")
                    print(f"  Cleaned: {len(cleaned)} chars")
                    print(f"  ANSI overhead: {reduction} chars ({reduction * 100 / len(output):.1f}%)")
                    print(f"  Has ANSI codes: {has_ansi}")

                    # Save individual agent outputs
                    filename = f"agent_{agent}.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(cleaned)
                    print(f"  Saved to: {filename}")

    except Exception as e:
        print(f"Error analyzing main endpoint: {e}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("Check the generated .txt files to see raw outputs")
    print("=" * 80)


if __name__ == "__main__":
    analyze_raw_output()