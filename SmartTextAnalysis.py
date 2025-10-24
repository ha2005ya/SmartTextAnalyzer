import os
import string
from collections import Counter, defaultdict
import pandas as pd

class TrieNode:
    def __init__(self):
        self.children = {}
        self.word_count = 0
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end = True
        current.word_count += 1

    def _dfs(self, node, prefix, results):
        if node.is_end:
            results.append((prefix, node.word_count))
        for char, child in node.children.items():
            self._dfs(child, prefix + char, results)

    def get_suggestions(self, prefix):
        current = self.root
        for char in prefix:
            if char not in current.children:
                return []
            current = current.children[char]
        results = []
        self._dfs(current, prefix, results)
        return sorted(results, key=lambda x: -x[1])[:5]

class SmartTextAnalyzer:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.translator = str.maketrans('', '', string.punctuation)
        self.cleaned_text, self.word_list = self.preprocess_text(raw_text)
        self.trie = Trie()
        self.undo_stack = []

        self.POSITIVE_WORDS = {"happy", "great", "love", "excellent", "good", "awesome", "fantastic", "nice", "like", "amazing"}
        self.NEGATIVE_WORDS = {"bad", "terrible", "sad", "hate", "worst", "awful", "angry", "disgusting", "horrible", "pain"}
        self.negations = {"not", "no", "never", "n't"}
        self.STOPWORDS = {"the", "is", "and", "of", "to", "in", "a", "that", "it", "on", "for", "with", "as", "at", "this", "an", "be", "by", "from", "or", "was", "are"}
        self.update_after_change()

        if os.path.exists("word.txt"):
            self.load_autocomplete_dataset("word.txt")
        else :
            print("Autocomplete dataset missing!")
        if os.path.exists("external_text_for_prediction.txt"):
            self.load_prediction_dataset("external_text_for_prediction.txt")
        else :
            print("External dataset missing")

    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(self.translator)
        words = text.split()
        cleaned_text = ' '.join(words)
        return cleaned_text, words

    def load_autocomplete_dataset(self , file_path):
        if not os.path.isfile(file_path):
            print(f"Dataset file '{file_path}' not found.")
            return
        with open(file_path , "r" , encoding = 'utf-8') as f :
            words = f.read().lower().translate(self.translator).split()
        for word in words :
            self.trie.insert(word)

    def load_prediction_dataset(self , path):
        if not os.path.isfile(path):
            print(f"Dataset file '{path}' not found.")
            return
        with open(path , "r" , encoding = 'utf-8') as f :
            words = f.read().lower().translate(self.translator).split()
            self.external_bigram = defaultdict(Counter)
            self.external_trigram = defaultdict(Counter)
            for i in range(len(words) - 1):
                self.external_bigram[words[i]][words[i+1]] += 1
            for i in range(len(words) - 2):
                self.external_trigram[(words[i],words[i+1])][words[i+2]] += 1

    def update_after_change(self):
        self.cleaned_text = ' '.join(self.word_list)
        self.trie = Trie()
        for word in self.word_list:
            self.trie.insert(word)
        self.bigram_dict = self.build_bigram_dict()
        self.trigram_dict = self.build_trigram_dict()
        self.cleaned_sentences = [s.strip().split() for s in self.cleaned_text.split(".") if s.strip()]

    def word_statistics(self):
        print("\n------ Word Statistics ------")
        print(f"Total words: {len(self.word_list)}")
        print(f"Unique words: {len(set(self.word_list))}")
        print("Top 5 frequent words:")
        freq = Counter(self.word_list)
        df = pd.DataFrame(freq.most_common(5),columns=["Word" , "Count"])
        print(df.to_string(index = False))

    def character_statistics(self):
        print("\n------ Character Statistics ------")
        text_no_spaces = self.cleaned_text.replace(" ", "")
        print(f"Total characters (excluding spaces): {len(text_no_spaces)}")
        print("Character frequencies:")
        freq = Counter(text_no_spaces)
        df = pd.DataFrame(freq.most_common(5) , columns=["Character" , "Count"])
        print(df.to_string(index=False))

    def search_word(self, word):
        results = []
        for i, sentence in enumerate(self.cleaned_sentences, 1):
            for j, w in enumerate(sentence, 1):
                if w.lower() == word:
                    results.append((i, j))
        if results:
            for sentence_no, word_no in results:
                print(f"Found in sentence {sentence_no}, word #{word_no}")
            print(f"Total occurrences: {len(results)}")
        else:
            print(f"The word '{word}' was not found.")

    def replace_word(self, old_word, new_word):
        changes = []
        for i in range(len(self.word_list)):
            if self.word_list[i] == old_word:
                changes.append((i , old_word))
                self.word_list[i] = new_word
        if changes :
            self.undo_stack.append(("replace" , changes))
            print(f"Replaced {len(changes)} occurrence(s) of '{old_word}' with '{new_word}'.")
            self.update_after_change()
        else :
            print(f"No occurrences of '{old_word}' were found.")

    def undo_last_change(self):
        if not self.undo_stack:
            print("No changes to undo.")
            return
        action = self.undo_stack.pop()
        if action[0] == "replace" :
            changes = action[1]
            for index , original_word in changes :
                self.word_list[index] = original_word
            self.update_after_change()
            print(f"Undo successful: Restored {len(changes)} word(s).")
        else :
            print("Unknown undo action.")
        self.update_after_change()
        print("Last change undone successfully.")

    def autocomplete(self, prefix):
        suggestions = self.trie.get_suggestions(prefix)
        if suggestions:
            print("Suggestions:")
            for i , (word , _) in enumerate(suggestions , 1):
                print(f"{i}. {word}")
        else:
            print("No suggestions found.")

    def build_bigram_dict(self):
        bigram_dict = defaultdict(Counter)
        for i in range(len(self.word_list) - 1):
            first, second = self.word_list[i], self.word_list[i + 1]
            bigram_dict[first][second] += 1
        return bigram_dict

    def build_trigram_dict(self):
        trigram_dict = defaultdict(Counter)
        for i in range(len(self.word_list) - 2):
            key = (self.word_list[i], self.word_list[i + 1])
            next_word = self.word_list[i + 2]
            trigram_dict[key][next_word] += 1
        return trigram_dict 

    def predict_next_word(self , word):
        model = getattr(self , "external_bigram" , self.bigram_dict )
        if word in model :
            predictions = model[word].most_common(5)
            print(f"Predictions after '{word}':")
            for i , (predicted_word , count) in enumerate(predictions , 1) :
                print(f"{i}. {predicted_word} ({count} times) ")
        else:
            print(f"No predictions found for '{word}'. ")

    def predict_next_word_trigram(self , phrase):
        words = phrase.lower().split()
        if len(words) != 2:
            print("Please enter exactly two words (separated by a space).")
            return
        key = (words[0], words[1])
        combined_counter = Counter()
        if hasattr(self , 'external_trigram') and key in self.external_trigram :
            combined_counter += self.external_trigram[key]
        if key in self.trigram_dict :
            combined_counter += self.trigram_dict[key]
        if combined_counter :
            predictions = combined_counter.most_common(5)
            print(f"Predictions after '{words[0]} {words[1]}': ")
            for i , (word , count) in enumerate(predictions , 1) :
                print(f"{i}. {word} ({count} times)")
        else :
            print(f"No trigram predictions found for '{words[0]} {words[1]}'.")

    def analyze_sentiment(self, sentence):
        sentence = sentence.translate(self.translator)
        words = sentence.split()

        pos_count = 0
        neg_count = 0
        skip_next = False
        for i in range(len(words)):
            if skip_next:
                skip_next = False
                continue
            word = words[i]
            if word in self.negations and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in self.POSITIVE_WORDS:
                    neg_count += 1
                    skip_next = True
                elif next_word in self.NEGATIVE_WORDS:
                    pos_count += 1
                    skip_next = True
            else:
                if word in self.POSITIVE_WORDS:
                    pos_count += 1
                elif word in self.NEGATIVE_WORDS:
                    neg_count += 1

        print(f"\nPositive words: {pos_count}, Negative words: {neg_count}")
        if pos_count > neg_count:
            print("Sentiment: 😊 Positive")
        elif neg_count > pos_count:
            print("Sentiment: 😡 Negative")
        else:
            print("Sentiment: 😐 Neutral")

    def extract_keywords(self):
        filtered_words = [word for word in self.word_list if word not in self.STOPWORDS and len(word) > 2]
        freq = Counter(filtered_words)
        top_keywords = freq.most_common(10)
        if not top_keywords:
            print("\nNo significant keywords found.")
            return
        print("\nTop Keywords (excluding stopwords):")
        for word, count in top_keywords:
            print(f"- {word}: {count} times")

    def generate_edits(self, word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def spell_suggestion(self , word):
        vocab = set(self.word_list)
        if word in vocab:
            print(f"'{word}' is found in the text.")
            return
        suggestions = [e for e in self.generate_edits(word) if e in vocab]
        if suggestions:
            print("Did you mean:")
            for s in sorted(suggestions)[:5]:  #top 5 suggestions max
                print(f"- {s}")
        else:
            print("No suggestions found.")

    def save_text_to_file(self , file_name):
        if not file_name:
            print("Invalid file name!")
            return
        file_path = file_name + ".txt"
        if os.path.exists(file_path):
            overwrite = input(f"'{file_path}' already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print('Save canceled.')
                return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.cleaned_text)
            print(f"Text saved successfully to '{file_path}'")
        except Exception as e:
            print(f"Error saving file: {e}")

    def show_menu(self):
        while True:
            print("""
---------- Main Menu ----------
1.\tWord statistics
2.\tCharacter statistics
3.\tSearch for a word
4.\tReplace a word
5.\tUndo last text modification
6.\tAutocomplete
7.\tNext word prediction
8.\tPredict next word (Trigram)
9.\tSentiment Analysis
10.\tExtract Keywords
11.\tSave modified text to file
12.\tSpell suggestion
13.\tExit""")
            choice = input("Choice: ").strip()
            if not choice.isdigit():
                print("Please enter a valid number!")
                continue

            if choice == "1":
                self.word_statistics()
            elif choice == "2":
                self.character_statistics()
            elif choice == "3":
                word = input("Enter the word to search: ").lower().strip()
                self.search_word(word)
            elif choice == "4":
                old = input("Enter the word to replace: ").lower().strip()
                new = input("Enter the new word: ").lower().strip()
                self.replace_word(old, new)
            elif choice == "5":
                self.undo_last_change()
            elif choice == "6":
                prefix = input("Enter the beginning of a word: ").lower().strip()
                self.autocomplete(prefix)
            elif choice == "7":
                word_to_predict = input("Enter a word to predict the next word: ").strip()
                self.predict_next_word(word_to_predict)
            elif choice == "8" :
                phrase = input("Enter two words to predict the next word: ").lower().strip()
                self.predict_next_word_trigram(phrase)
            elif choice == "9":
                sentence = input("Enter a sentence to analyze sentiment: ").strip()
                self.analyze_sentiment(sentence)
            elif choice == "10":
                self.extract_keywords()
            elif choice == "11":
                file_name = input("Enter file name to save (without extension): ").strip()
                self.save_text_to_file(file_name)
            elif choice == "12":
                word_to_spell = input("Enter a word to check: ").lower().strip()
                self.spell_suggestion(word_to_spell)
            elif choice == "13":
                save_prompt = input("Do you want to save your changes before exiting? (y/n): ").strip().lower()
                if save_prompt == "y":
                    file_name = input("Enter file name to save (without extension): ").strip()
                    self.save_text_to_file(file_name)
                print("\nThank you for using Smart Text Analyzer. Goodbye!")
                break
            else:
                print("Invalid choice!")


def load_text():
    print("Welcome to the Intelligent Text Processor!")
    choice = input("Do you want to load from a file (f) or enter it directly (d)? [F/D]: ").strip().lower()
    text = ""
    if choice == 'f':
        file_path = input("Please enter the full path to your text file: ").strip()
        if not os.path.isfile(file_path):
            print("File does not exist.")
            return ""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif choice == 'd':
        print("Enter your text. Type '$$end_text$$' in a new line to finish:")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == '$$end_text$$':
                break
            lines.append(line)
        text = '\n'.join(lines)
        if not text.strip():
            print("No text loaded. Try again...\n")
    else:
        print("Invalid choice.")
    return text


# Main Program
while True:
    print()
    raw_text = load_text()
    analyzer = SmartTextAnalyzer(raw_text)
    if not raw_text :
        break
    analyzer.show_menu()