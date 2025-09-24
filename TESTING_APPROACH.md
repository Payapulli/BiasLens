# BiasLens Unit Testing Approach

## Overview

This document outlines the unit testing strategy implemented for the BiasLens political bias analysis system. The focus is on testing individual components in isolation to ensure reliability and maintainability.

## Testing Philosophy

### 1. **Component Isolation**
- Test each component independently
- Mock external dependencies (DistilGPT2, FAISS, sentence transformers)
- Use realistic test data that mirrors production scenarios

### 2. **Error Handling Focus**
- Test both success and failure scenarios
- Verify graceful degradation when components fail
- Ensure proper error messages and fallback behavior

### 3. **Simplicity and Maintainability**
- Keep tests focused and readable
- Use clear test names and structure
- Minimize test dependencies

## Test Structure

```
tests/
├── __init__.py
├── test_retriever.py      # DocumentRetriever tests
├── test_distilgpt2.py     # DistilGPT2 analysis tests
└── test_server.py         # FastAPI endpoint tests
```

## Component Testing

### 1. DocumentRetriever Tests (`test_retriever.py`)

**Purpose**: Test document retrieval and FAISS indexing functionality

**Key Test Cases**:
- Initialization with valid/invalid files
- Document retrieval with various query types
- Error handling for missing files and malformed data
- Edge cases (empty queries, invalid JSON)

**Mocking Strategy**:
- Mock `sentence_transformers.SentenceTransformer` for embeddings
- Mock `faiss.IndexFlatIP` for vector search
- Use temporary files for realistic data testing

### 2. DistilGPT2 Analysis Tests (`test_distilgpt2.py`)

**Purpose**: Test DistilGPT2 response parsing and analysis logic

**Key Test Cases**:
- Structured response parsing (BIAS, CONFIDENCE, EVIDENCE, RATIONALE)
- Confidence value normalization (percentages to decimals)
- Bias label normalization and validation
- Evidence extraction and word limiting
- Error handling for malformed responses
- Prompt building with and without few-shot examples

**Mocking Strategy**:
- Mock `transformers.AutoTokenizer` and `AutoModelForCausalLM`
- Simulate various response formats and error conditions
- Test confidence value edge cases (0-1 range, percentages)

### 3. FastAPI Server Tests (`test_server.py`)

**Purpose**: Test API endpoints and request/response handling

**Key Test Cases**:
- Health check endpoint
- Analysis endpoint with various inputs
- Error handling for invalid requests
- Response structure validation
- Special character and large query handling

**Mocking Strategy**:
- Mock all heavy components (retriever, DistilGPT2)
- Use `TestClient` for HTTP testing
- Simulate various error conditions

## Testing Data Strategy

### 1. **Realistic Test Articles**
```json
{
  "title": "Test Article 1",
  "content": "This is test content 1",
  "bias_label": "leans_left",
  "source": "test"
}
```

### 2. **Edge Case Testing**
- Empty queries
- Very long queries
- Special characters and emojis
- Malformed JSON data
- Missing required fields

## Error Handling Testing

### 1. **Component Failures**
- Retriever initialization errors
- DistilGPT2 loading failures
- FAISS index corruption
- File I/O errors

### 2. **Data Validation**
- Invalid input formats
- Malformed responses
- Out-of-range values

## Test Execution

### Running Tests
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_retriever.py
```

### Test Quality Metrics
- **Coverage**: Focus on critical paths and error handling
- **Reliability**: Tests should be deterministic and not flaky
- **Maintainability**: Clear test structure and minimal dependencies

## Key Testing Principles

### 1. **Mock Heavy Components**
- DistilGPT2 and sentence transformers are mocked to avoid GPU requirements
- FAISS operations are mocked for consistent testing
- File I/O is mocked or uses temporary files

### 2. **Test Error Paths**
- Every component has error handling tests
- Verify graceful degradation when components fail
- Ensure proper error messages and fallback behavior

### 3. **Realistic Test Data**
- Use test data that mirrors production scenarios
- Test with various input types and edge cases
- Ensure tests cover the most common use cases

## Future Considerations

### 1. **Integration Testing**
- Add integration tests for component interactions
- Test the full pipeline with realistic data
- Performance testing for response times

### 2. **Coverage Monitoring**
- Add code coverage reporting
- Set coverage thresholds for critical components
- Monitor test coverage over time

## Conclusion

This unit testing approach ensures that the BiasLens system components are reliable and maintainable. By testing each component in isolation with proper mocking and realistic data, we can confidently deploy the system while maintaining high quality standards.

The testing strategy balances thoroughness with simplicity, focusing on the most critical components while ensuring that the system behaves correctly under both normal and error conditions.