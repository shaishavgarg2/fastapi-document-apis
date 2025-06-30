from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import google.generativeai as genai
import base64
import json
import re
from difflib import SequenceMatcher
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class DocumentInput(BaseModel):
    fileContent: str = Field(..., description="Base64 encoded protobuf content")
    fileExtension: str = Field(..., description="File extension (e.g., .pdf, .txt)")
    fileName: str = Field(..., description="Original file name")

class RequestParameter(BaseModel):
    parameterName: str = Field(..., description="Name of the parameter to find across documents")
    parameterValue: str = Field(..., description="Expected value to match")

class RelationMatchRequest(BaseModel):
    documents: List[DocumentInput] = Field(..., description="List of documents to analyze")
    requestParameters: List[RequestParameter] = Field(..., description="Parameters to match across documents")

class ParameterResult(BaseModel):
    parameterName: str
    parameterValue: str
    foundParameterValue: Optional[str] = None
    nearestHeading: Optional[str] = None
    isValueFound: bool = False
    pageNumber: Optional[int] = None
    lineNumber: Optional[int] = None
    matchingAccuracyPercentage: float = 0.0
    fileName: str

class DocumentRelationResult(BaseModel):
    parameterName: str
    responseParameter: List[ParameterResult]

class RelationMatchResponse(BaseModel):
    isSuccess: bool
    findDocumentRelation: List[DocumentRelationResult]
    relationshipSummary: Optional[Dict[str, Any]] = None
    errorMessage: Optional[str] = None

class DocumentRelationshipAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the analyzer with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def decode_protobuf_content(self, base64_content: str) -> str:
        """Decode base64 protobuf content to readable text"""
        try:
            decoded_bytes = base64.b64decode(base64_content)
            
            try:
                text_content = decoded_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_content = decoded_bytes.decode('latin-1')
                except UnicodeDecodeError:
                    text_content = decoded_bytes.decode('utf-8', errors='ignore')
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error decoding protobuf content: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to decode document content: {str(e)}")
    
    def preprocess_document(self, document: DocumentInput) -> Dict[str, Any]:
        """Preprocess document content to extract structure information"""
        text_content = self.decode_protobuf_content(document.fileContent)
        lines = text_content.split('\n')
        processed_lines = []
        headings = []
        
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            if stripped_line:
                # Enhanced heading detection for financial documents
                if (stripped_line.isupper() and len(stripped_line) > 3) or \
                   re.match(r'^[A-Z][A-Z\s]{5,}$', stripped_line) or \
                   re.match(r'^\d+\.\s+[A-Z]', stripped_line) or \
                   any(keyword in stripped_line.upper() for keyword in [
                       'INVOICE', 'SHIPPING', 'DEPOSIT', 'STATEMENT', 'TOTAL', 'SUMMARY', 
                       'BILLING', 'PAYMENT', 'TRANSACTION', 'AMOUNT DUE'
                   ]):
                    headings.append({
                        'text': stripped_line,
                        'line_number': i,
                        'type': 'heading'
                    })
                
                processed_lines.append({
                    'line_number': i,
                    'content': stripped_line,
                    'original_line': line
                })
        
        return {
            'lines': processed_lines,
            'headings': headings,
            'total_lines': len(lines),
            'text_content': text_content,
            'document_info': {
                'fileName': document.fileName,
                'fileExtension': document.fileExtension
            }
        }
    
    def find_nearest_heading(self, target_line: int, headings: List[Dict]) -> Optional[str]:
        """Find the nearest heading above the target line"""
        nearest_heading = None
        min_distance = float('inf')
        
        for heading in headings:
            if heading['line_number'] <= target_line:
                distance = target_line - heading['line_number']
                if distance < min_distance:
                    min_distance = distance
                    nearest_heading = heading['text']
        
        return nearest_heading
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity percentage between two strings with enhanced financial matching"""
        # Normalize strings for comparison
        text1_norm = re.sub(r'[^\w\d.]', '', text1.lower())
        text2_norm = re.sub(r'[^\w\d.]', '', text2.lower())
        
        # Use SequenceMatcher for basic similarity
        basic_similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        # Enhanced matching for financial data
        enhanced_similarity = basic_similarity
        
        # Amount matching enhancement
        if '$' in text1 or '$' in text2:
            amount1 = self.extract_amount(text1)
            amount2 = self.extract_amount(text2)
            if amount1 and amount2:
                if abs(amount1 - amount2) < 0.01:  # Within 1 cent
                    enhanced_similarity = max(enhanced_similarity, 0.98)
                elif abs(amount1 - amount2) / max(amount1, amount2) < 0.01:  # Within 1%
                    enhanced_similarity = max(enhanced_similarity, 0.95)
        
        # Date matching enhancement
        date1 = self.extract_date(text1)
        date2 = self.extract_date(text2)
        if date1 and date2:
            if date1 == date2:
                enhanced_similarity = max(enhanced_similarity, 0.96)
        
        return round(enhanced_similarity * 100, 1)
    
    def extract_amount(self, text: str) -> Optional[float]:
        """Extract monetary amount from text"""
        # Match patterns like $1,234.56, $1234, 1,234.56, etc.
        patterns = [
            r'\$?([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*(?:USD|dollars?)',
            r'(?:amount|total|sum)[\s:]*\$?([\d,]+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    amount_str = match.group(1).replace(',', '')
                    return float(amount_str)
                except ValueError:
                    continue
        return None
    
    def extract_date(self, text: str) -> Optional[str]:
        """Extract and normalize date from text"""
        # Various date patterns
        date_patterns = [
            r'(\d{1,2}[\s/\-]\w+[\s/\-]\d{4})',  # 15 June 2024
            r'(\w+[\s/\-]\d{1,2}[\s/\-]\d{4})',  # June 15 2024
            r'(\d{1,2}[\s/\-]\d{1,2}[\s/\-]\d{4})',  # 06/15/2024
            r'(\d{4}[\s/\-]\d{1,2}[\s/\-]\d{1,2})',  # 2024-06-15
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None
    
    def search_parameter_in_document(self, parameter_name: str, parameter_value: str, 
                                   processed_doc: Dict) -> ParameterResult:
        """Search for a parameter in a single document using enhanced AI and direct search"""
        
        # Try direct search first
        direct_result = self.search_parameter_direct(parameter_name, parameter_value, processed_doc)
        
        if direct_result['found'] and direct_result['accuracy'] >= 90:
            return ParameterResult(
                parameterName=parameter_name,
                parameterValue=parameter_value,
                foundParameterValue=direct_result['found_value'],
                nearestHeading=direct_result['nearest_heading'],
                isValueFound=True,
                pageNumber=1,
                lineNumber=direct_result['line_number'],
                matchingAccuracyPercentage=direct_result['accuracy'],
                fileName=processed_doc['document_info']['fileName']
            )
        
        # Use AI search for more complex matching
        ai_result = self.search_parameter_with_ai(parameter_name, parameter_value, processed_doc)
        
        line_number = None
        nearest_heading = None
        
        if ai_result.get('found') and ai_result.get('context_line'):
            context_line = ai_result['context_line']
            for line_data in processed_doc['lines']:
                if context_line.lower() in line_data['content'].lower():
                    line_number = line_data['line_number']
                    nearest_heading = self.find_nearest_heading(line_number, processed_doc['headings'])
                    break
        
        return ParameterResult(
            parameterName=parameter_name,
            parameterValue=parameter_value,
            foundParameterValue=ai_result.get('found_value'),
            nearestHeading=nearest_heading,
            isValueFound=ai_result.get('found', False),
            pageNumber=1,
            lineNumber=line_number,
            matchingAccuracyPercentage=ai_result.get('confidence_score', 0),
            fileName=processed_doc['document_info']['fileName']
        )
    
    def search_parameter_direct(self, parameter_name: str, parameter_value: str, 
                              processed_doc: Dict) -> Dict[str, Any]:
        """Direct text search for parameter value with financial document awareness"""
        lines = processed_doc['lines']
        headings = processed_doc['headings']
        
        best_match = {
            'found': False,
            'line_number': None,
            'found_value': None,
            'accuracy': 0,
            'context_line': None,
            'nearest_heading': None
        }
        
        # Search through all lines
        for line_data in lines:
            line_content = line_data['content']
            line_number = line_data['line_number']
            
            # Enhanced matching for financial data
            similarity = self.calculate_similarity(parameter_value, line_content)
            
            # Check for exact match
            if parameter_value.lower() in line_content.lower():
                accuracy = 100
                found_value = parameter_value
                
                # Look for the exact substring in the line
                start_idx = line_content.lower().find(parameter_value.lower())
                if start_idx != -1:
                    end_idx = start_idx + len(parameter_value)
                    found_value = line_content[start_idx:end_idx]
                
                best_match = {
                    'found': True,
                    'line_number': line_number,
                    'found_value': found_value,
                    'accuracy': accuracy,
                    'context_line': line_content,
                    'nearest_heading': self.find_nearest_heading(line_number, headings)
                }
                break
            
            # Check for high similarity matches
            elif similarity > best_match['accuracy'] and similarity > 70:
                # Extract potential match from the line
                found_value = self.extract_best_match(parameter_value, line_content)
                
                best_match = {
                    'found': True,
                    'line_number': line_number,
                    'found_value': found_value or line_content.strip(),
                    'accuracy': similarity,
                    'context_line': line_content,
                    'nearest_heading': self.find_nearest_heading(line_number, headings)
                }
        
        return best_match
    
    def extract_best_match(self, target: str, line: str) -> Optional[str]:
        """Extract the best matching portion from a line"""
        # For amounts, extract the amount from the line
        if '$' in target or any(word in target.lower() for word in ['amount', 'total', 'sum']):
            amount_match = re.search(r'\$?[\d,]+\.?\d*', line)
            if amount_match:
                return amount_match.group(0)
        
        # For dates, extract the date from the line
        if any(word in target.lower() for word in ['date', 'june', 'january', 'february', 'march', 'april', 'may', 'july', 'august', 'september', 'october', 'november', 'december']) or re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{4}', target):
            date_match = self.extract_date(line)
            if date_match:
                return date_match
        
        # For other matches, find the most similar word or phrase
        words = line.split()
        best_word = max(words, key=lambda w: SequenceMatcher(None, target.lower(), w.lower()).ratio())
        return best_word if SequenceMatcher(None, target.lower(), best_word.lower()).ratio() > 0.6 else None
    
    def search_parameter_with_ai(self, parameter_name: str, parameter_value: str, 
                               processed_doc: Dict) -> Dict[str, Any]:
        """Use AI to find parameter in document with enhanced context for financial documents"""
        
        text_content = processed_doc['text_content']
        file_name = processed_doc['document_info']['fileName']
        
        # Determine document type for better AI prompting
        doc_type = self.determine_document_type(file_name, text_content)
        
        prompt = f"""
        Analyze this {doc_type} document and find the parameter "{parameter_name}" with value "{parameter_value}".
        
        Document content:
        {text_content[:8000]}  # Limit content to avoid token limits
        
        Search for:
        - Parameter Name: "{parameter_name}"
        - Expected Value: "{parameter_value}"
        
        For financial documents, consider:
        - Amount variations: $75,500.00 = $75500 = 75,500 USD
        - Date variations: 15 June,2024 = 06/15/2024 = 2024-06-15
        - Context clues: Invoice numbers, reference numbers, transaction IDs
        - Financial headings: Total Due, Amount Payable, Final Summary
        
        Return in this exact JSON format:
        {{
            "found": true/false,
            "found_value": "exact text found in document or null",
            "context_line": "full line where the value was found or null",
            "estimated_line_number": number or null,
            "confidence_score": percentage (0-100),
            "nearby_context": "surrounding text that provides context",
            "reasoning": "explanation of how the match was found",
            "document_type": "{doc_type}"
        }}
        
        Be precise and only return valid JSON.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON response
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()
            
            # Find JSON boundaries
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end]
                ai_result = json.loads(response_text)
                return ai_result
            else:
                logger.warning("No valid JSON found in AI response")
                return {"found": False, "confidence_score": 0}
                
        except Exception as e:
            logger.error(f"AI search error: {str(e)}")
            return {"found": False, "confidence_score": 0, "error": str(e)}
    
    def determine_document_type(self, file_name: str, content: str) -> str:
        """Determine the type of financial document"""
        file_name_lower = file_name.lower()
        content_lower = content.lower()
        
        if 'invoice' in file_name_lower or 'invoice' in content_lower:
            return 'invoice'
        elif 'shipping' in file_name_lower or 'shipping' in content_lower or 'delivery' in content_lower:
            return 'shipping document'
        elif 'deposit' in file_name_lower or 'deposit' in content_lower:
            return 'deposit listing'
        elif 'statement' in file_name_lower or 'bank' in content_lower or 'account' in content_lower:
            return 'bank statement'
        else:
            return 'financial document'
    
    def analyze_document_relationships(self, results: List[DocumentRelationResult]) -> Dict[str, Any]:
        """Analyze relationships and patterns across documents"""
        relationship_summary = {
            'total_parameters': len(results),
            'total_matches': 0,
            'cross_document_matches': {},
            'document_coverage': {},
            'potential_discrepancies': [],
            'relationship_strength': 0.0
        }
        
        for param_result in results:
            param_name = param_result.parameterName
            found_docs = [r for r in param_result.responseParameter if r.isValueFound]
            
            relationship_summary['total_matches'] += len(found_docs)
            relationship_summary['cross_document_matches'][param_name] = {
                'found_in_documents': len(found_docs),
                'document_names': [r.fileName for r in found_docs],
                'values_found': [r.foundParameterValue for r in found_docs],
                'accuracy_scores': [r.matchingAccuracyPercentage for r in found_docs]
            }
            
            # Check for discrepancies
            if len(found_docs) > 1:
                values = [r.foundParameterValue for r in found_docs]
                if len(set(values)) > 1:  # Different values found
                    relationship_summary['potential_discrepancies'].append({
                        'parameter': param_name,
                        'conflicting_values': values,
                        'documents': [r.fileName for r in found_docs]
                    })
        
        # Calculate document coverage
        all_documents = set()
        for param_result in results:
            for doc_result in param_result.responseParameter:
                all_documents.add(doc_result.fileName)
        
        for doc_name in all_documents:
            matches_in_doc = sum(1 for param_result in results 
                               for doc_result in param_result.responseParameter 
                               if doc_result.fileName == doc_name and doc_result.isValueFound)
            relationship_summary['document_coverage'][doc_name] = {
                'parameters_found': matches_in_doc,
                'coverage_percentage': round((matches_in_doc / len(results)) * 100, 1)
            }
        
        # Calculate overall relationship strength
        total_possible_matches = len(results) * len(all_documents)
        actual_matches = sum(len([r for r in param_result.responseParameter if r.isValueFound]) 
                           for param_result in results)
        relationship_summary['relationship_strength'] = round((actual_matches / total_possible_matches) * 100, 1)
        
        return relationship_summary
    
    def find_document_relations(self, request: RelationMatchRequest) -> RelationMatchResponse:
        """Main method to find relationships across multiple documents"""
        try:
            # Preprocess all documents
            processed_documents = []
            for document in request.documents:
                logger.info(f"Processing document: {document.fileName}")
                processed_doc = self.preprocess_document(document)
                processed_documents.append(processed_doc)
            
            # Search for each parameter across all documents
            relation_results = []
            
            for param in request.requestParameters:
                logger.info(f"Searching for parameter: {param.parameterName} = {param.parameterValue}")
                
                parameter_results = []
                
                # Search in each document
                for processed_doc in processed_documents:
                    result = self.search_parameter_in_document(
                        param.parameterName, 
                        param.parameterValue, 
                        processed_doc
                    )
                    parameter_results.append(result)
                
                relation_results.append(DocumentRelationResult(
                    parameterName=param.parameterName,
                    responseParameter=parameter_results
                ))
            
            # Analyze relationships
            relationship_summary = self.analyze_document_relationships(relation_results)
            
            return RelationMatchResponse(
                isSuccess=True,
                findDocumentRelation=relation_results,
                relationshipSummary=relationship_summary
            )
            
        except Exception as e:
            logger.error(f"Error processing document relations: {str(e)}")
            return RelationMatchResponse(
                isSuccess=False,
                findDocumentRelation=[],
                errorMessage=str(e)
            )

# Global analyzer instance
analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global analyzer
    
    # Startup
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set!")
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    analyzer = DocumentRelationshipAnalyzer(api_key)
    logger.info("DocumentRelationshipAnalyzer initialized successfully")
    
    yield  # This is where the app runs
    
    # Shutdown (if needed)
    logger.info("Application shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Document Relationship Matching API",
    description="API to find relationships and matches across multiple financial documents using AI",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/api/find-document-relations", response_model=RelationMatchResponse)
async def find_document_relations(request: RelationMatchRequest):
    """
    Find relationships and matches across multiple documents
    
    - **documents**: List of documents to analyze (protobuf format, base64 encoded)
    - **requestParameters**: Parameters to find and match across all documents
    
    Returns detailed cross-document matching results with relationship analysis.
    """
    if analyzer is None:
        raise HTTPException(status_code=500, detail="Service not initialized properly")
    
    try:
        result = analyzer.find_document_relations(request)
        return result
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Relationship Matching API",
        "version": "1.0.0",
        "endpoints": {
            "find_document_relations": "/api/find-document-relations",
            "health": "/health",
            "docs": "/docs"
        },
        "supported_use_cases": [
            "Invoice ↔ Shipping Document Matching",
            "Deposit Listing ↔ Bank Statement Matching", 
            "Multiple Deposit Aggregation",
            "Full 4-Document Cross-Verification"
        ]
    }

# Example usage and testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)