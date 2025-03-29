# attribute_quality.py

import spacy  # pip install spacy && python -m spacy download en_core_web_sm
from lexical_diversity import lex_div as ld 

nlp = spacy.load("en_core_web_sm")

class AttributeQualityChecker:
    """Assess the quality of metadata attribute values beyond basic validation."""

    def __init__(self):
        self.min_keywords = 3
        self.lexical_diversity_threshold = 0.5  # Higher = More diverse
        self.keyword_similarity_threshold = 1 # 1 = All keywords must be unique
        self.sentence_variety_threshold = 2  # Minimum unique sentence structures
        self.mattr_window = 20  # Window size for MATTR calculation

    def check_attribute_quality(self, attribute_name, value):
        """Check the quality of a given attribute's value."""

        if attribute_name == "description":
            return self.check_description(value)

        if attribute_name == "keywords":
            return self.check_keywords(value)

        return True, "attribute quality is acceptable."

    def check_description(self, value):
        """Assess the quality of the dataset description."""
        lexical_diversity = self.calculate_lexical_diversity(value)
        sentence_variety = self.calculate_sentence_variety(value)

        if lexical_diversity < self.lexical_diversity_threshold:
            return False, "The description lacks lexical diversity and may be repetitive."
        if sentence_variety < self.sentence_variety_threshold:
            return False, "The description has limited sentence variety and may be monotonous."

        return True, "Description quality is acceptable."
    
    def calculate_lexical_diversity(self, value):
        """Calculate the lexical diversity (Word Uniqueness) using MATTR."""
        words = value.split()
        if len(words) > self.mattr_window: 
            mattr_score = ld.mattr(words, window_length=self.mattr_window)
        else:
            mattr_score = 0
        return mattr_score

    def calculate_sentence_variety(self, value):
        """Calculate the variety of sentence structures in the description."""
        sentence_structures = [self.get_sentence_structure(sent) for sent in nlp(value).sents]
        unique_structures = len(set(sentence_structures))
        return unique_structures

    def check_keywords(self, value):
        """Ensure keywords are diverse and not redundant."""
        keyword_list = [kw.strip().lower() for kw in value.split(",") if kw.strip()]
        unique_keywords = set(keyword_list)
        
        # Check minimum number of keywords
        if len(keyword_list) < self.min_keywords:
            return False, f"Please provide at least {self.min_keywords} keywords."

        # Check keyword uniqueness
        if len(unique_keywords) / len(keyword_list) < self.keyword_similarity_threshold:
            return False, "Keywords should not have any repeated words."
        
        return True, "Keywords quality is acceptable."

    def get_sentence_structure(self, sentence):
        """Extracts a simplified structure of a sentence (e.g., 'NOUN VERB NOUN')."""
        return " ".join([token.pos_ for token in sentence])

    def check_quality_of_all_attributes(self, metadata):
        """Evaluate all metadata attributes and return quality issues."""
        issues = {}
        for attribute_name, value in metadata.items():
            is_valid, message = self.check_attribute_quality(attribute_name, value)
            if not is_valid:
                issues[attribute_name] = message
        return issues