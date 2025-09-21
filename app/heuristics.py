"""
Bias detection heuristics using keyword analysis and sentiment.
"""

import re
from typing import Dict, List, Any, Tuple
from textblob import TextBlob


class BiasHeuristics:
    """Simple heuristics for detecting political bias in text."""
    
    def __init__(self):
        """Initialize bias detection heuristics."""
        # Left-leaning keywords and phrases
        self.left_keywords = [
            'corporate greed', 'wealthy elite', 'working families', 'social justice',
            'income inequality', 'climate crisis', 'systemic racism', 'progressive',
            'universal healthcare', 'free college', 'minimum wage', 'union rights',
            'environmental protection', 'renewable energy', 'affordable housing',
            'social safety net', 'wealth tax', 'corporate accountability',
            'workers rights', 'climate action', 'green new deal', 'medicare for all'
        ]
        
        # Right-leaning keywords and phrases
        self.right_keywords = [
            'free market', 'individual liberty', 'small government', 'fiscal responsibility',
            'traditional values', 'law and order', 'personal responsibility', 'entrepreneurship',
            'deregulation', 'tax cuts', 'economic freedom', 'constitutional rights',
            'national security', 'border security', 'family values', 'religious freedom',
            'prosperity', 'job creators', 'innovation', 'competition', 'self-reliance',
            'limited government', 'free enterprise', 'patriotism', 'conservative'
        ]
        
        # Neutral/center indicators
        self.neutral_keywords = [
            'bipartisan', 'compromise', 'balanced approach', 'evidence-based',
            'data-driven', 'pragmatic', 'moderate', 'centrist', 'nonpartisan',
            'objective', 'factual', 'unbiased', 'fair', 'equitable'
        ]
        
        # Emotional/loaded language patterns
        self.emotional_patterns = [
            r'\b(unprecedented|historic|devastating|catastrophic|revolutionary)\b',
            r'\b(always|never|all|every|none|nothing)\b',  # Absolute language
            r'\b(obviously|clearly|undoubtedly|certainly)\b',  # Certainty markers
            r'\b(disaster|crisis|emergency|breakthrough|miracle)\b'  # Dramatic language
        ]
    
    def analyze_keywords(self, text: str) -> Dict[str, Any]:
        """Analyze text for bias-related keywords."""
        text_lower = text.lower()
        
        # Count keyword occurrences
        left_count = sum(1 for keyword in self.left_keywords if keyword in text_lower)
        right_count = sum(1 for keyword in self.right_keywords if keyword in text_lower)
        neutral_count = sum(1 for keyword in self.neutral_keywords if keyword in text_lower)
        
        # Calculate keyword density
        total_words = len(text.split())
        left_density = left_count / max(total_words, 1) * 100
        right_density = right_count / max(total_words, 1) * 100
        neutral_density = neutral_count / max(total_words, 1) * 100
        
        # Find specific keywords used
        found_left = [kw for kw in self.left_keywords if kw in text_lower]
        found_right = [kw for kw in self.right_keywords if kw in text_lower]
        found_neutral = [kw for kw in self.neutral_keywords if kw in text_lower]
        
        return {
            'left_count': left_count,
            'right_count': right_count,
            'neutral_count': neutral_count,
            'left_density': left_density,
            'right_density': right_density,
            'neutral_density': neutral_density,
            'found_left': found_left,
            'found_right': found_right,
            'found_neutral': found_neutral
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and emotional language."""
        blob = TextBlob(text)
        
        # Basic sentiment
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Emotional language detection
        emotional_score = 0
        emotional_phrases = []
        
        for pattern in self.emotional_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                emotional_score += len(matches)
                emotional_phrases.extend(matches)
        
        # Normalize emotional score
        total_words = len(text.split())
        emotional_density = emotional_score / max(total_words, 1) * 100
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'emotional_score': emotional_score,
            'emotional_density': emotional_density,
            'emotional_phrases': emotional_phrases
        }
    
    def analyze_language_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze language patterns that might indicate bias."""
        # Absolute language detection
        absolute_patterns = [
            r'\b(always|never|all|every|none|nothing|no one|everyone)\b',
            r'\b(obviously|clearly|undoubtedly|certainly|definitely)\b',
            r'\b(proven|established|fact|truth)\b'
        ]
        
        absolute_count = 0
        for pattern in absolute_patterns:
            absolute_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Question patterns (might indicate uncertainty or challenge)
        question_count = text.count('?')
        
        # Exclamation patterns (might indicate strong emotion)
        exclamation_count = text.count('!')
        
        # Length and complexity
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        return {
            'absolute_language_count': absolute_count,
            'question_count': question_count,
            'exclamation_count': exclamation_count,
            'avg_sentence_length': avg_sentence_length,
            'sentence_count': len(sentences)
        }
    
    def calculate_bias_score(self, text: str) -> Dict[str, Any]:
        """
        Calculate overall bias score and classification.
        
        Returns:
            Dictionary with bias analysis results
        """
        # Run all analyses
        keyword_analysis = self.analyze_keywords(text)
        sentiment_analysis = self.analyze_sentiment(text)
        pattern_analysis = self.analyze_language_patterns(text)
        
        # Calculate bias score (-1 to 1, where -1 is left, 0 is center, 1 is right)
        left_score = keyword_analysis['left_density'] / 100
        right_score = keyword_analysis['right_density'] / 100
        neutral_score = keyword_analysis['neutral_density'] / 100
        
        # Weight the scores
        keyword_bias = (right_score - left_score) / max(left_score + right_score + 0.1, 1)
        
        # Factor in sentiment (positive sentiment with right keywords = more right bias)
        sentiment_factor = sentiment_analysis['polarity'] * keyword_bias * 0.3
        
        # Factor in emotional language (more emotional = potentially more biased)
        emotional_factor = min(sentiment_analysis['emotional_density'] / 10, 1) * keyword_bias * 0.2
        
        # Combine scores
        raw_bias_score = keyword_bias + sentiment_factor + emotional_factor
        bias_score = max(-1, min(1, raw_bias_score))  # Clamp to [-1, 1]
        
        # Determine tentative label
        if abs(bias_score) < 0.2:
            tentative_label = "center"
        elif bias_score > 0.2:
            tentative_label = "leans_right"
        else:
            tentative_label = "leans_left"
        
        # Calculate confidence based on strength of signals
        confidence = min(abs(bias_score) + neutral_score / 100, 1.0)
        
        return {
            'bias_score': bias_score,
            'tentative_label': tentative_label,
            'confidence': confidence,
            'keyword_analysis': keyword_analysis,
            'sentiment_analysis': sentiment_analysis,
            'pattern_analysis': pattern_analysis,
            'raw_scores': {
                'left_score': left_score,
                'right_score': right_score,
                'neutral_score': neutral_score,
                'keyword_bias': keyword_bias,
                'sentiment_factor': sentiment_factor,
                'emotional_factor': emotional_factor
            }
        }
    
    def get_bias_indicators(self, text: str) -> List[str]:
        """Get specific indicators of bias found in the text."""
        indicators = []
        
        keyword_analysis = self.analyze_keywords(text)
        sentiment_analysis = self.analyze_sentiment(text)
        pattern_analysis = self.analyze_language_patterns(text)
        
        # Keyword-based indicators
        if keyword_analysis['left_count'] > 0:
            indicators.append(f"Left-leaning keywords: {', '.join(keyword_analysis['found_left'][:3])}")
        
        if keyword_analysis['right_count'] > 0:
            indicators.append(f"Right-leaning keywords: {', '.join(keyword_analysis['found_right'][:3])}")
        
        if keyword_analysis['neutral_count'] > 0:
            indicators.append(f"Neutral language: {', '.join(keyword_analysis['found_neutral'][:3])}")
        
        # Sentiment-based indicators
        if sentiment_analysis['polarity'] > 0.3:
            indicators.append("Strong positive sentiment")
        elif sentiment_analysis['polarity'] < -0.3:
            indicators.append("Strong negative sentiment")
        
        if sentiment_analysis['subjectivity'] > 0.7:
            indicators.append("High subjectivity")
        
        # Pattern-based indicators
        if pattern_analysis['absolute_language_count'] > 2:
            indicators.append("Use of absolute language")
        
        if sentiment_analysis['emotional_density'] > 5:
            indicators.append("High emotional language")
        
        return indicators


def main():
    """Test the heuristics module."""
    heuristics = BiasHeuristics()
    
    # Test texts
    test_texts = [
        "The economy has experienced unprecedented growth this quarter, with GDP rising 4.2% and unemployment dropping to historic lows.",
        "Corporate greed continues to exploit working families while the wealthy hoard resources that should be shared with the community.",
        "The bipartisan infrastructure bill represents a compromise that addresses critical national needs while respecting fiscal constraints."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        print("-" * 80)
        
        result = heuristics.calculate_bias_score(text)
        print(f"Bias Score: {result['bias_score']:.3f}")
        print(f"Label: {result['tentative_label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        indicators = heuristics.get_bias_indicators(text)
        print(f"Indicators: {', '.join(indicators)}")
        print()


if __name__ == "__main__":
    main()
