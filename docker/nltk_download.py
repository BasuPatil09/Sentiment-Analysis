import nltk
import sys
import time

packages = ["stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"]
dest = sys.argv[1]

for pkg in packages:
    for attempt in range(1, 4):
        try:
            nltk.download(pkg, download_dir=dest, quiet=True, raise_on_error=True)
            break
        except Exception as e:
            print(f"[nltk-download] {pkg} attempt {attempt}/3 failed: {e}")
            if attempt == 3:
                sys.exit(1)
            time.sleep(3)
