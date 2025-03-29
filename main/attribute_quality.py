# attribute_quality.py

# necessary imports
import spacy  
from lexical_diversity import lex_div as ld 

nlp = spacy.load("en_core_web_sm")

class AttributeQualityChecker:
    """A class to assess the quality of metadata attribute values beyond basic validation."""

    def __init__(self):
        self.min_keywords = 3
        self.lexical_diversity_threshold = 0.5  # Higher = More diverse
        self.keyword_similarity_threshold = 1 # 1 = All keywords must be unique
        self.sentence_variety_threshold = 2  # Minimum unique sentence structures
        self.mattr_window = 20  # Window size for MATTR calculation

    def check_attribute_quality(self, attribute_name: str, value: str) -> tuple[bool, str]:
        """
        Check the quality of a given attribute's value.

        Args:
            attribute_name: The name of the attribute (e.g., 'description', 'keywords').
            value: The value of the attribute to be checked.

        Returns:
            A tuple containing a boolean indicating quality and a message.
        """
        # Check if the attribute is description or keywords
        if attribute_name == "description":
            return self.check_description(value)

        if attribute_name == "keywords":
            return self.check_keywords(value)

        return True, "Attribute quality is acceptable."

    def check_description(self, value: str) -> tuple[bool, str]:
        """
        Assess the quality of the dataset description.

        Args:
            value: The description text to be evaluated.

        Returns:
            A tuple containing a boolean indicating quality and a message.

        """
        try:
            # Calculate lexical diversity and sentence variety, handling errors
            lexical_diversity, lex_error = self.calculate_lexical_diversity(value)
            if lex_error:
                return False, lex_error
            sentence_variety, sen_error = self.calculate_sentence_variety(value)
            if sen_error:
                return False, sen_error
            
            # Check lexical diversity and sentence variety against thresholds
            if lexical_diversity < self.lexical_diversity_threshold:
                return False, "The description lacks lexical diversity and may be repetitive."
            if sentence_variety < self.sentence_variety_threshold:
                return False, "The description has limited sentence variety and may be monotonous."

            return True, "Description quality is acceptable."
        except Exception as e:
            return False, f"Error in description quality check: {str(e)}"
    
    def calculate_lexical_diversity(self, value: str) -> tuple[float, str | None]:
        """
        Calculate the lexical diversity (Word Uniqueness) using MATTR.

        Args:
            value: The text for which lexical diversity is to be calculated.

        Returns:
            A tuple where the first element is the MATTR score (0 if calculation fails),
            and the second element is an error message (None if no error).
        """
        try:
            words = value.split() # Tokenize the text into words
            # Calculate MATTR (Mean Type-Token Ratio) for lexical diversity
            if len(words) > self.mattr_window: 
                mattr_score = ld.mattr(words, window_length=self.mattr_window)
            else:
                mattr_score = 0
            return mattr_score, None
        except Exception as e:
            return 0, f"Error in lexical diversity calculation: {str(e)}"

    def calculate_sentence_variety(self, value: str) -> tuple[int, str | None]:
        """
        Calculate the variety of sentence structures in the description.

        Args:
            value: The text for which sentence variety is to be calculated.

        Returns:
            A tuple where the first element is the count of unique sentence structures,
            and the second element is an error message (None if no error).
        """
        try:
            sentence_structures = []
            for sent in nlp(value).sents:
                structure, error = self.get_sentence_structure(sent)
                if error:
                    return 0, f"Error in sentence variety calculation: {error}"  # Return the error immediately
                sentence_structures.append(structure)

            unique_structures = len(set(sentence_structures))  # Count unique sentence structures
            return unique_structures, None  # No error
        except Exception as e:
            return 0, f"Error in sentence variety calculation: {str(e)}"
 
    def get_sentence_structure(self, sentence) -> tuple[str, str | None]:
        """
        Extract a simplified structure of a sentence (e.g., 'NOUN VERB NOUN').

        Args:
            sentence: A spaCy sentence object to analyze.

        Returns:
            A tuple where the first element is the sentence structure as a string,
            and the second element is an error message (None if no error).
        """
        try:
            return " ".join([token.pos_ for token in sentence]), None
        except Exception as e:
            return "", f"Error in extracting sentence structure: {str(e)}"

    def check_keywords(self, value: str) -> tuple[bool, str]:
        """
        Ensure keywords are diverse and not redundant.

        Args:
            value: A comma-separated string of keywords to be evaluated.

        Returns:
            A tuple where the first element indicates if the keywords quality is acceptable,
            and the second element provides a message or reason.
        """
        try:
            keyword_list = [kw.strip().lower() for kw in value.split(",") if kw.strip()]
            unique_keywords = set(keyword_list)
            
            # Check minimum number of keywords
            if len(keyword_list) < self.min_keywords:
                return False, f"Please provide at least {self.min_keywords} keywords."

            # Check keyword uniqueness
            if len(unique_keywords) / len(keyword_list) < self.keyword_similarity_threshold:
                return False, "Keywords should not have any repeated words."
            
            return True, "Keywords quality is acceptable."
        except Exception as e:
            return False, f"Error in keywords quality check: {str(e)}"

    def check_quality_of_all_attributes(self, metadata: dict[str, str]) -> dict[str, str]:
        """
        Evaluate all metadata attributes and return quality issues.

        Args:
            metadata: A dictionary where keys are attribute names and values are their respective values.

        Returns:
            A dictionary of issues where keys are attribute names and values are the respective quality issues.
        """
        try:
            issues = {}
            for attribute_name, value in metadata.items():
                is_valid, message = self.check_attribute_quality(attribute_name, value)
                if not is_valid:
                    issues[attribute_name] = message
            return issues
        except Exception as e:
            return {"error": f"Error in checking quality of all attributes: {str(e)}"}