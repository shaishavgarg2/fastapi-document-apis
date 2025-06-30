from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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

# Initialize FastAPI app
app = FastAPI(
    title="Document Parameter Matching API",
    description="API to match specific parameter values in documents using AI",
    version="1.0.0"
)

# Pydantic models for request/response
class RequestParameter(BaseModel):
    parameterName: str = Field(..., description="Name of the parameter to find")
    parameterValue: str = Field(..., description="Expected value to match")

class Document(BaseModel):
    fileContent: str = Field(..., description="Base64 encoded protobuf content")
    fileExtension: str = Field(..., description="File extension (e.g., .txt, .proto)")
    fileName: str = Field(..., description="Original file name")

class MatchRequest(BaseModel):
    requestParameters: List[RequestParameter] = Field(..., description="Parameters to match")
    document: Document = Field(..., description="Document to search in")

class ResponseParameter(BaseModel):
    parameterName: str
    parameterValue: str
    foundParameterValue: Optional[str] = None
    nearestHeading: Optional[str] = None
    isValueFound: bool = False
    pageNumber: Optional[int] = None
    lineNumber: Optional[int] = None
    matchingAccuracyPercentage: int = 0
    fileName: str

class MatchResponse(BaseModel):
    isSuccess: bool
    responseParameter: List[ResponseParameter]
    errorMessage: Optional[str] = None

class DocumentParameterMatcher:
    def __init__(self, api_key: str):
        """Initialize the matcher with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
    
    def decode_protobuf_content(self, base64_content: str) -> str:
        """Decode base64 protobuf content to readable text"""
        try:
            # Decode base64 to bytes
            decoded_bytes = base64.b64decode(base64_content)
            
            # Convert bytes to string (assuming UTF-8 encoding)
            # Note: For actual protobuf parsing, you'd need the .proto schema
            # This is a simplified approach treating it as text content
            try:
                text_content = decoded_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try other encodings
                try:
                    text_content = decoded_bytes.decode('latin-1')
                except UnicodeDecodeError:
                    text_content = decoded_bytes.decode('utf-8', errors='ignore')
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error decoding protobuf content: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to decode document content: {str(e)}")
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Preprocess text content to extract structure information"""
        lines = text.split('\n')
        processed_lines = []
        headings = []
        
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            if stripped_line:
                # Simple heading detection (lines that are all caps, or have specific patterns)
                if (stripped_line.isupper() and len(stripped_line) > 3) or \
                   re.match(r'^[A-Z][A-Z\s]{5,}$', stripped_line) or \
                   re.match(r'^\d+\.\s+[A-Z]', stripped_line):
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
            'total_lines': len(lines)
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
        """Calculate similarity percentage between two strings"""
        # Normalize strings for comparison
        text1_norm = re.sub(r'[^\w\d.]', '', text1.lower())
        text2_norm = re.sub(r'[^\w\d.]', '', text2.lower())
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()
        return round(similarity * 100)
    
    def search_parameter_with_ai(self, parameter_name: str, parameter_value: str, 
                                text_content: str, processed_data: Dict) -> Dict[str, Any]:
        """Use AI to find parameter in document with enhanced context"""
        
        prompt = f"""
        Analyze the following document text and find the parameter "{parameter_name}" with value "{parameter_value}".
        
        Document content:
        {text_content[:8000]}  # Limit content to avoid token limits
        
        Search for:
        - Parameter Name: "{parameter_name}"
        - Expected Value: "{parameter_value}"
        
        Look for exact matches, partial matches, or semantically similar content.
        
        Return in this exact JSON format:
        {{
            "found": true/false,
            "found_value": "exact text found in document or null",
            "context_line": "full line where the value was found or null",
            "estimated_line_number": number or null,
            "confidence_score": percentage (0-100),
            "nearby_context": "surrounding text that provides context",
            "reasoning": "explanation of how the match was found"
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
    
    def search_parameter_direct(self, parameter_name: str, parameter_value: str, 
                              processed_data: Dict) -> Dict[str, Any]:
        """Direct text search for parameter value"""
        lines = processed_data['lines']
        headings = processed_data['headings']
        
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
            
            # Check for partial matches (fuzzy matching)
            words_in_line = line_content.split()
            for word in words_in_line:
                similarity = self.calculate_similarity(parameter_value, word)
                if similarity > best_match['accuracy'] and similarity > 70:  # 70% threshold
                    best_match = {
                        'found': True,
                        'line_number': line_number,
                        'found_value': word,
                        'accuracy': similarity,
                        'context_line': line_content,
                        'nearest_heading': self.find_nearest_heading(line_number, headings)
                    }
        
        return best_match
    
    def match_parameters(self, request: MatchRequest) -> MatchResponse:
        """Main method to match parameters in document"""
        try:
            # Decode document content
            text_content = self.decode_protobuf_content(request.document.fileContent)
            
            # Preprocess text
            processed_data = self.preprocess_text(text_content)
            
            response_parameters = []
            
            for param in request.requestParameters:
                logger.info(f"Searching for parameter: {param.parameterName} = {param.parameterValue}")
                
                # Try direct search first
                direct_result = self.search_parameter_direct(
                    param.parameterName, 
                    param.parameterValue, 
                    processed_data
                )
                
                if direct_result['found'] and direct_result['accuracy'] >= 90:
                    # Use direct search result if high confidence
                    response_param = ResponseParameter(
                        parameterName=param.parameterName,
                        parameterValue=param.parameterValue,
                        foundParameterValue=direct_result['found_value'],
                        nearestHeading=direct_result['nearest_heading'],
                        isValueFound=True,
                        pageNumber=1,  # For text documents, treat as single page
                        lineNumber=direct_result['line_number'],
                        matchingAccuracyPercentage=direct_result['accuracy'],
                        fileName=request.document.fileName
                    )
                else:
                    # Use AI search for more complex matching
                    ai_result = self.search_parameter_with_ai(
                        param.parameterName,
                        param.parameterValue,
                        text_content,
                        processed_data
                    )
                    
                    # Determine line number from AI result
                    line_number = None
                    nearest_heading = None
                    
                    if ai_result.get('found') and ai_result.get('context_line'):
                        # Try to find the line number by searching for the context
                        context_line = ai_result['context_line']
                        for line_data in processed_data['lines']:
                            if context_line.lower() in line_data['content'].lower():
                                line_number = line_data['line_number']
                                nearest_heading = self.find_nearest_heading(line_number, processed_data['headings'])
                                break
                    
                    response_param = ResponseParameter(
                        parameterName=param.parameterName,
                        parameterValue=param.parameterValue,
                        foundParameterValue=ai_result.get('found_value'),
                        nearestHeading=nearest_heading,
                        isValueFound=ai_result.get('found', False),
                        pageNumber=1,
                        lineNumber=line_number,
                        matchingAccuracyPercentage=ai_result.get('confidence_score', 0),
                        fileName=request.document.fileName
                    )
                
                response_parameters.append(response_param)
            
            return MatchResponse(
                isSuccess=True,
                responseParameter=response_parameters
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return MatchResponse(
                isSuccess=False,
                responseParameter=[],
                errorMessage=str(e)
            )

# Global matcher instance (in production, use dependency injection)
matcher = None

@app.on_event("startup")
async def startup_event():
    """Initialize the matcher with API key on startup"""
    global matcher
    
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set!")
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    matcher = DocumentParameterMatcher(api_key)
    logger.info("DocumentParameterMatcher initialized successfully")

@app.post("/api/match-parameters", response_model=MatchResponse)
async def match_parameters(request: MatchRequest):
    """
    Match specific parameter values in a document
    
    - **requestParameters**: List of parameters to find with their expected values
    - **document**: Document content in protobuf format (base64 encoded)
    
    Returns detailed matching results with location and accuracy information.
    """
    if matcher is None:
        raise HTTPException(status_code=500, detail="Service not initialized properly")
    
    try:
        result = matcher.match_parameters(request)
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
        "message": "Document Parameter Matching API",
        "version": "1.0.0",
        "endpoints": {
            "match_parameters": "/api/match-parameters",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Example usage and testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)