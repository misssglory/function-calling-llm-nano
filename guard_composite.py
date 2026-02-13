"""
Complete Testing Script for Russian LLM Guard with Tiny Local Models
Runs entirely locally without any external API calls
"""

import os
import sys
import json
import re
import time
import unittest
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Install required packages if not present
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("Installing required packages...")
    os.system("pip install numpy scikit-learn sentence-transformers torch")
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    import torch

# ==================== CONFIGURATION ====================


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Action(Enum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


@dataclass
class GuardResult:
    text: str
    is_safe: bool
    risk_level: RiskLevel
    action: Action
    violations: Dict
    redacted_text: Optional[str] = None
    processing_time: float = 0.0
    model_used: str = "hybrid"


# ==================== TINY LOCAL MODELS ====================


class TinyRussianRegexGuard:
    """Lightweight regex-based guard for Russian text"""

    def __init__(self):
        # Russian personal data patterns
        self.pii_patterns = {
            "passport": [
                r"\b\d{4}\s?\d{6}\b",  # Russian passport
                r"\b\d{2}\s?\d{2}\s?\d{6}\b",  # Alternative format
            ],
            "phone": [
                r"\+7[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",
                r"8[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",
            ],
            "snils": [
                r"\b\d{3}[-]?\d{3}[-]?\d{3}[-]?\d{2}\b",
            ],
            "inn": [
                r"\b\d{10}\b|\b\d{12}\b",
            ],
            "email": [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            ],
        }

        # Russian toxic patterns (simplified for testing)
        self.toxic_patterns = {
            "profanity": [
                r"\b[бв]ля\w*",
                r"\bх[ую]й\w*",
                r"\bп[ие]зд\w*",
                r"\b[её]ба\w*",
                r"\bмудак\w*",
            ],
            "threats": [
                r"\bуб[ьи]ю\b",
                r"\bприконч[уи]\b",
                r"\bвзо[рв][а-я]+\b",
            ],
            "insults": [
                r"\bдурак\b",
                r"\bидиот\b",
                r"\bдебил\b",
            ],
        }

        # Compile all patterns
        self.compiled_patterns = {}
        for category, patterns in {**self.pii_patterns, **self.toxic_patterns}.items():
            self.compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def check(self, text: str) -> Dict:
        """Check text against all regex patterns"""
        violations = {}

        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(text)
                if found:
                    matches.extend(found)
            if matches:
                violations[category] = matches

        return violations

    def redact(self, text: str, violations: Dict) -> str:
        """Redact PII from text"""
        redacted = text

        # Only redact PII categories
        pii_categories = ["passport", "phone", "snils", "inn", "email"]

        for category in pii_categories:
            if category in violations:
                for match in violations[category]:
                    redacted = redacted.replace(match, f"[{category.upper()}_REDACTED]")

        return redacted


class TinyRussianEmbeddingGuard:
    """Lightweight embedding-based guard using tiny sentence transformers"""

    def __init__(self):
        # Load tiny multilingual model (only 80MB)
        print("Loading tiny embedding model (80MB)...")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # Define blocked topics in Russian
        self.blocked_topics = {
            "hate_speech": [
                "я ненавижу людей другой национальности",
                "всех нужно уничтожить",
                "эта нация хуже других",
            ],
            "extremism": [
                "призывы к насилию",
                "организация незаконных акций",
                "террористические действия",
            ],
            "personal_info_request": [
                "дай свой паспорт",
                "сообщи номер карты",
                "скажи снилс",
                "напиши свои паспортные данные",
            ],
            "drug_content": [
                "купить наркотики",
                "продажа запрещенных веществ",
                "изготовление наркотиков",
            ],
        }

        # Pre-compute embeddings
        print("Pre-computing topic embeddings...")
        self.topic_embeddings = {}
        for topic, examples in self.blocked_topics.items():
            self.topic_embeddings[topic] = self.model.encode(examples)

        self.similarity_threshold = 0.65  # Adjustable threshold

    def check(self, text: str) -> Dict:
        """Check text similarity to blocked topics"""
        if not text or len(text.strip()) == 0:
            return {}

        # Encode input text
        text_embedding = self.model.encode([text])

        violations = {}

        for topic, topic_embs in self.topic_embeddings.items():
            # Calculate similarities
            similarities = cosine_similarity(text_embedding, topic_embs)
            max_similarity = np.max(similarities)

            if max_similarity > self.similarity_threshold:
                violations[topic] = float(max_similarity)

        return violations


class TinyRussianLLMGuard:
    """
    Ultra-lightweight LLM guard using a tiny distilled model
    For testing, we'll use a rule-based fallback since tiny LLMs are still large
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

        if not use_mock:
            try:
                # Try to load tiny DistilBERT for Russian
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSequenceClassification,
                )

                print("Loading tiny LLM guard model...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "DeepPavlov/distilrubert-tiny-cased-conversational"
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "DeepPavlov/distilrubert-tiny-cased-conversational", num_labels=2
                )
                self.use_mock = False
            except Exception as e:
                print(f"Could not load model, using mock: {e}")
                self.use_mock = True

        # Mock patterns for testing
        self.mock_patterns = [
            (r"(?:угроза|опасность|смерть|убить)", "threat", 0.8),
            (r"(?:пароль|логин|аккаунт)", "account_security", 0.6),
            (r"(?:кредит|деньги|перевод)", "financial", 0.5),
        ]

    def analyze(self, text: str) -> Dict:
        """Analyze text for complex content violations"""
        if self.use_mock:
            return self._mock_analyze(text)
        else:
            return self._model_analyze(text)

    def _mock_analyze(self, text: str) -> Dict:
        """Mock analysis for testing"""
        results = {}

        for pattern, category, score in self.mock_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                results[category] = score

        return results

    def _model_analyze(self, text: str) -> Dict:
        """Actual model inference"""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

        # Assuming binary classification (toxic/not toxic)
        toxic_prob = probs[0][1].item()

        if toxic_prob > 0.5:
            return {"toxic_content": toxic_prob}
        return {}


# ==================== HYBRID GUARD ====================


class RussianHybridGuard:
    """Combines all guard approaches for comprehensive protection"""

    def __init__(self, use_tiny_models: bool = True):
        print("Initializing Russian Hybrid Guard with tiny models...")

        # Initialize all guards
        self.regex_guard = TinyRussianRegexGuard()
        self.embedding_guard = TinyRussianEmbeddingGuard()

        # LLM guard is optional (can be heavy)
        self.llm_guard = None
        if use_tiny_models:
            try:
                self.llm_guard = TinyRussianLLMGuard(
                    use_mock=True
                )  # Start with mock for testing
            except:
                pass

        # Risk thresholds
        self.thresholds = {
            RiskLevel.HIGH: 0.7,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.LOW: 0.0,
        }

        print("✓ Hybrid Guard initialized successfully")

    def analyze(self, text: str) -> GuardResult:
        """Complete analysis of text"""
        start_time = time.time()

        # Step 1: Regex check (fastest)
        regex_violations = self.regex_guard.check(text)

        # Step 2: Embedding check (semantic)
        embedding_violations = self.embedding_guard.check(text)

        # Step 3: LLM check (if available)
        llm_violations = {}
        if self.llm_guard:
            llm_violations = self.llm_guard.analyze(text)

        # Combine all violations
        all_violations = {
            "regex": regex_violations,
            "semantic": embedding_violations,
            "llm": llm_violations,
        }

        # Calculate risk score
        risk_score = self._calculate_risk_score(
            regex_violations, embedding_violations, llm_violations
        )

        # Determine risk level and action
        risk_level, action = self._determine_risk_and_action(
            risk_score, regex_violations
        )

        # Redact if needed
        redacted_text = None
        if action in [Action.FLAG, Action.BLOCK]:
            redacted_text = self.regex_guard.redact(text, regex_violations)

        processing_time = time.time() - start_time

        return GuardResult(
            text=text,
            is_safe=action == Action.ALLOW,
            risk_level=risk_level,
            action=action,
            violations=all_violations,
            redacted_text=redacted_text,
            processing_time=processing_time,
            model_used="tiny-hybrid",
        )

    def _calculate_risk_score(
        self, regex_violations: Dict, embedding_violations: Dict, llm_violations: Dict
    ) -> float:
        """Calculate overall risk score"""
        score = 0.0

        # Weight regex violations (PII is high risk)
        if regex_violations:
            for category in regex_violations:
                if category in ["passport", "snils", "inn"]:
                    score += 0.4
                elif category in ["phone", "email"]:
                    score += 0.3
                elif category in ["profanity", "threats"]:
                    score += 0.5

        # Weight embedding violations (semantic matches)
        if embedding_violations:
            for category, similarity in embedding_violations.items():
                score += similarity * 0.3

        # Weight LLM violations
        if llm_violations:
            for _, value in llm_violations.items():
                score += value * 0.2

        return min(score, 1.0)

    def _determine_risk_and_action(
        self, risk_score: float, regex_violations: Dict
    ) -> Tuple[RiskLevel, Action]:
        """Determine risk level and required action"""

        # Check for immediate block conditions
        if any(cat in regex_violations for cat in ["threats", "extremism"]):
            return RiskLevel.HIGH, Action.BLOCK

        if "passport" in regex_violations or "snils" in regex_violations:
            return RiskLevel.HIGH, Action.FLAG

        # Use risk score
        if risk_score >= self.thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH, Action.BLOCK
        elif risk_score >= self.thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM, Action.FLAG
        else:
            return RiskLevel.LOW, Action.ALLOW


# ==================== TEST SUITE ====================


class TestRussianHybridGuard(unittest.TestCase):
    """Comprehensive test suite for Russian hybrid guard"""

    @classmethod
    def setUpClass(cls):
        """Initialize once for all tests"""
        print("\n" + "=" * 60)
        print("Setting up Russian Hybrid Guard Test Suite")
        print("=" * 60)
        cls.guard = RussianHybridGuard(use_tiny_models=True)

    def test_01_safe_text(self):
        """Test with completely safe Russian text"""
        print("\n📝 Test 1: Safe Text")
        texts = [
            "Привет, как дела? Хорошая погода сегодня.",
            "Можно заказать пиццу на ужин?",
            "Сегодня в 15:00 будет собрание.",
            "Спасибо за помощь в решении задачи.",
        ]

        for text in texts:
            result = self.guard.analyze(text)
            print(f"  Input: {text[:30]}...")
            print(
                f"  Safe: {result.is_safe}, Risk: {result.risk_level.value}, Time: {result.processing_time:.3f}s"
            )
            self.assertTrue(result.is_safe)
            self.assertEqual(result.action, Action.ALLOW)

    def test_02_pii_detection(self):
        """Test detection of Russian PII"""
        print("\n📝 Test 2: PII Detection")
        test_cases = [
            ("Мой паспорт: 1234 567890", "passport"),
            ("Телефон: +7 (999) 123-45-67", "phone"),
            ("СНИЛС: 123-456-789 01", "snils"),
            ("ИНН: 1234567890", "inn"),
            ("Email: test@example.com", "email"),
        ]

        for text, expected_category in test_cases:
            result = self.guard.analyze(text)
            print(f"  Input: {text}")
            print(f"  Detected: {list(result.violations['regex'].keys())}")
            self.assertFalse(result.is_safe)
            self.assertIn(expected_category, result.violations["regex"])
            if result.redacted_text:
                print(f"  Redacted: {result.redacted_text}")

    def test_03_profanity_detection(self):
        """Test detection of Russian profanity"""
        print("\n📝 Test 3: Profanity Detection")
        profane_texts = [
            "Ты настоящий дурак!",
            "Это полный дебилизм.",
            "Отстань, придурок!",
        ]

        for text in profane_texts:
            result = self.guard.analyze(text)
            print(f"  Input: {text}")
            print(f"  Safe: {result.is_safe}, Action: {result.action.value}")
            if result.violations["regex"]:
                print(f"  Found: {list(result.violations['regex'].keys())}")
            self.assertEqual(result.action, Action.FLAG)

    def test_04_semantic_detection(self):
        """Test semantic detection using embeddings"""
        print("\n📝 Test 4: Semantic Detection")
        test_cases = [
            ("Я хочу купить запрещенные вещества", "drug_content"),
            ("Ненавижу всех людей другой национальности", "hate_speech"),
            ("Давай устроим теракт", "extremism"),
        ]

        for text, expected_topic in test_cases:
            result = self.guard.analyze(text)
            print(f"  Input: {text}")
            if result.violations["semantic"]:
                print(f"  Semantic matches: {result.violations['semantic']}")
            self.assertGreater(len(result.violations["semantic"]), 0)

    def test_05_mixed_content(self):
        """Test with mixed safe and unsafe content"""
        print("\n📝 Test 5: Mixed Content")
        texts = [
            "Привет! Мой телефон +7 999 123-45-67, позвони мне",
            "Сегодня хорошая погода, но этот идиот все испортил",
            "Паспортные данные: 1234 567890. Остальное нормально",
        ]

        for text in texts:
            result = self.guard.analyze(text)
            print(f"  Input: {text}")
            print(f"  Safe: {result.is_safe}, Risk: {result.risk_level.value}")
            if result.redacted_text:
                print(f"  Redacted: {result.redacted_text}")
            self.assertFalse(result.is_safe)

    def test_06_edge_cases(self):
        """Test edge cases"""
        print("\n📝 Test 6: Edge Cases")
        edge_cases = [
            ("", "empty string"),
            ("   ", "whitespace only"),
            ("Тест" * 1000, "very long text"),
            ("12345", "just numbers"),
            ("Привет! :)", "with emoji"),
        ]

        for text, description in edge_cases:
            result = self.guard.analyze(text)
            print(f"  Case: {description}")
            print(f"  Safe: {result.is_safe}, Time: {result.processing_time:.3f}s")
            self.assertIsNotNone(result)

    def test_07_redaction_quality(self):
        """Test quality of PII redaction"""
        print("\n📝 Test 7: Redaction Quality")
        text = "Мои данные: паспорт 1234 567890, телефон +7 999 123-45-67, email user@test.com"
        result = self.guard.analyze(text)

        print(f"  Original: {text}")
        print(f"  Redacted: {result.redacted_text}")

        # Check that sensitive data is removed
        self.assertNotIn("1234 567890", result.redacted_text)
        self.assertNotIn("+7 999 123-45-67", result.redacted_text)
        self.assertNotIn("user@test.com", result.redacted_text)

        # Check that redaction markers are present
        self.assertIn("[PASSPORT_REDACTED]", result.redacted_text)
        self.assertIn("[PHONE_REDACTED]", result.redacted_text)
        self.assertIn("[EMAIL_REDACTED]", result.redacted_text)

    def test_08_performance_benchmark(self):
        """Test performance on multiple texts"""
        print("\n📝 Test 8: Performance Benchmark")

        # Generate test texts
        test_texts = [
            "Привет, как дела?" * 10,  # 10x repetition
            "Мой телефон +7 999 123-45-67" * 5,
            "Этот текст содержит обычные слова и фразы на русском языке",
            "Опасный контент с угрозами и плохими словами",
        ] * 5  # 20 total texts

        print(f"  Testing {len(test_texts)} texts...")

        start_time = time.time()
        results = [self.guard.analyze(text) for text in test_texts]
        total_time = time.time() - start_time

        avg_time = total_time / len(test_texts)
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per text: {avg_time:.3f}s")
        print(f"  Throughput: {1/avg_time:.1f} texts/second")

        self.assertLess(avg_time, 1.0)  # Should be under 1 second per text

    def test_09_accuracy_metrics(self):
        """Calculate accuracy metrics"""
        print("\n📝 Test 9: Accuracy Metrics")

        # Test cases with expected results
        test_cases = [
            ("Привет, мир!", True),  # Safe
            ("Ты дурак!", False),  # Unsafe (insult)
            ("+7 999 1234567", False),  # Unsafe (phone)
            ("Хорошая погода", True),  # Safe
            ("Нужно купить наркотики", False),  # Unsafe (drugs)
            ("Сегодня пятница", True),  # Safe
            ("Убью тебя", False),  # Unsafe (threat)
            ("Спасибо за помощь", True),  # Safe
        ]

        correct = 0
        total = len(test_cases)

        print("  Detailed results:")
        for text, expected_safe in test_cases:
            result = self.guard.analyze(text)
            is_correct = result.is_safe == expected_safe
            correct += 1 if is_correct else 0

            status = "✓" if is_correct else "✗"
            print(f"  {status} Input: {text[:30]}")
            print(f"    Expected safe: {expected_safe}, Got: {result.is_safe}")

        accuracy = correct / total
        print(f"\n  Accuracy: {accuracy:.2%} ({correct}/{total} correct)")

        self.assertGreaterEqual(accuracy, 0.7)  # At least 70% accuracy

    def test_10_batch_processing(self):
        """Test batch processing capability"""
        print("\n📝 Test 10: Batch Processing")

        batch_size = 10
        texts = [
            f"Тестовый текст {i} с телефоном +7 999 123-45-67"
            for i in range(batch_size)
        ]

        print(f"  Processing batch of {batch_size} texts...")

        # Sequential processing
        start_seq = time.time()
        seq_results = [self.guard.analyze(text) for text in texts]
        seq_time = time.time() - start_seq

        # Simulate parallel (just for comparison)
        start_par = time.time()
        par_results = [
            self.guard.analyze(text) for text in texts
        ]  # Actually same, just for demo
        par_time = time.time() - start_par

        print(f"  Sequential time: {seq_time:.3f}s")
        print(f"  Average per text: {seq_time/batch_size:.3f}s")

        # Verify all results
        for i, result in enumerate(seq_results):
            self.assertIsNotNone(result)
            if i == 0:
                print(f"  Sample result for text {i}: Action={result.action.value}")


# ==================== MAIN EXECUTION ====================


def run_demo():
    """Run a demonstration of the guard"""
    print("\n" + "=" * 60)
    print("Russian LLM Guard - Interactive Demo")
    print("=" * 60)

    guard = RussianHybridGuard(use_tiny_models=True)

    demo_texts = [
        "Привет! Меня зовут Александр.",
        "Мой номер телефона: +7 (999) 123-45-67",
        "Ты полный дурак и идиот!",
        "Паспортные данные: 4510 123456",
        "Давай купим запрещенные вещества",
        "Хорошая погода сегодня, не правда ли?",
    ]

    print("\n📊 Demo Results:")
    print("-" * 80)
    print(f"{'Text':<30} {'Safe':<8} {'Risk':<8} {'Action':<10} {'Time(ms)':<10}")
    print("-" * 80)

    for text in demo_texts:
        result = guard.analyze(text)

        # Truncate long text
        display_text = text[:27] + "..." if len(text) > 30 else text

        print(
            f"{display_text:<30} {str(result.is_safe):<8} {result.risk_level.value:<8} "
            f"{result.action.value:<10} {result.processing_time*1000:.1f}"
        )

        if result.violations["regex"] or result.violations["semantic"]:
            print(f"  {' ':<30} Violations: {result.violations['regex']}")
            if result.violations["semantic"]:
                print(f"  {' ':<30} Semantic: {result.violations['semantic']}")

    print("-" * 80)


def run_tests():
    """Run all unit tests"""
    print("\n" + "=" * 60)
    print("Running Complete Test Suite")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRussianHybridGuard)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


def export_results_to_json(
    results: List[GuardResult], filename: str = "guard_results.json"
):
    """Export test results to JSON"""
    data = []
    for r in results:
        data.append(
            {
                "text": r.text,
                "is_safe": r.is_safe,
                "risk_level": r.risk_level.value,
                "action": r.action.value,
                "violations": r.violations,
                "redacted_text": r.redacted_text,
                "processing_time": r.processing_time,
            }
        )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Results exported to {filename}")


def main():
    """Main function"""
    print("\n" + "★" * 60)
    print("Russian LLM Guard - Complete Testing Script")
    print("★" * 60)

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check available packages
    print("\n📦 Package versions:")
    print(f"  numpy: {np.__version__}")
    print(f"  sentence-transformers: {SentenceTransformer.__name__}")
    print(f"  torch: {torch.__version__ if 'torch' in dir() else 'Not installed'}")

    # Menu
    while True:
        print("\n" + "=" * 60)
        print("Select option:")
        print("1. Run quick demo")
        print("2. Run full test suite")
        print("3. Benchmark performance")
        print("4. Interactive mode")
        print("5. Export results to JSON")
        print("0. Exit")
        print("=" * 60)

        choice = input("Enter choice (0-5): ").strip()

        if choice == "1":
            run_demo()

        elif choice == "2":
            result = run_tests()
            print(
                f"\nTest Summary: {result.testsRun} tests run, "
                f"{len(result.failures)} failures, "
                f"{len(result.errors)} errors"
            )

        elif choice == "3":
            print("\n📊 Running benchmark...")
            guard = RussianHybridGuard(use_tiny_models=True)

            import time

            texts = [
                f"Тестовый текст {i} с номером +7 999 123-45-67" for i in range(50)
            ]

            start = time.time()
            for text in texts:
                guard.analyze(text)
            total = time.time() - start

            print(f"Processed 50 texts in {total:.2f} seconds")
            print(f"Average: {total/50*1000:.2f} ms per text")
            print(f"Throughput: {50/total:.1f} texts/second")

        elif choice == "4":
            print("\n💬 Interactive Mode (type 'quit' to exit)")
            guard = RussianHybridGuard(use_tiny_models=True)

            while True:
                text = input("\nEnter Russian text: ").strip()
                if text.lower() in ["quit", "exit", "q"]:
                    break

                result = guard.analyze(text)
                print(f"\nResults:")
                print(f"  Safe: {result.is_safe}")
                print(f"  Risk Level: {result.risk_level.value}")
                print(f"  Action: {result.action.value}")
                print(f"  Processing time: {result.processing_time*1000:.2f} ms")

                if result.violations["regex"]:
                    print(f"  Regex violations: {result.violations['regex']}")
                if result.violations["semantic"]:
                    print(f"  Semantic violations: {result.violations['semantic']}")
                if result.redacted_text:
                    print(f"  Redacted: {result.redacted_text}")

        elif choice == "5":
            print("\n📁 Exporting results...")
            guard = RussianHybridGuard(use_tiny_models=True)

            test_texts = [
                "Привет, мир!",
                "Мой телефон +7 999 123-45-67",
                "Ты дурак!",
                "Паспорт 1234 567890",
            ]

            results = [guard.analyze(text) for text in test_texts]
            export_results_to_json(results)

        elif choice == "0":
            print("\nExiting. До свидания!")
            break

        else:
            print("Invalid choice, please try again")


if __name__ == "__main__":
    main()
