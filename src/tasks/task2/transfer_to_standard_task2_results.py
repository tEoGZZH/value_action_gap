import pandas as pd
import json
import argparse

def to_t2_standard(eval_csv: str) -> pd.DataFrame:
    df = pd.read_csv(eval_csv)

    long_rows = []
    for _, r in df.iterrows():
        country, topic, value = r["country"], r["topic"], r["value"]

        for i in range(8):
            cell = r.get(f"evaluation_{i}")
            if pd.isna(cell):
                continue

            # Parse JSON
            if isinstance(cell, str):
                try:
                    obj = json.loads(cell)
                except Exception:
                    obj = eval(cell)
            else:
                obj = cell

            action = obj.get("action")  # "Option 1" / "Option 2"
            if action not in ("Option 1", "Option 2"):
                continue

            # Option1=negative, Option2=positive
            chosen_polarity = "negative" if action == "Option 1" else "positive"

            # Create two rows for each evaluation, one for negative polarity and one for positive polarity
            for polarity in ("negative", "positive"):
                long_rows.append({
                    "country": country,
                    "topic": topic,
                    "value": value,
                    "prompt_index": str(i),
                    "polarity": polarity,
                    "model_choice": (polarity == chosen_polarity),
                })

    return pd.DataFrame(long_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", type=str)
    parser.add_argument("--out_csv", type=str)
    args = parser.parse_args()

    long_df = to_t2_standard(args.eval_csv)
    long_df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
