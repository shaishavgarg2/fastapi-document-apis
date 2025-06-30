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
class Document(BaseModel):
    fileContent: str = Field(..., description="Base64 encoded protobuf content")
    fileExtension: str = Field(..., description="File extension (e.g., .pdf, .txt)")
    fileName: str = Field(..., description="Original file name")

class ValueExtractionRequest(BaseModel):
    document: Document = Field(..., description="Document to extract values from")
    parameterNames: List[str] = Field(..., description="List of parameter names to find values for")

class ExtractedParameter(BaseModel):
    parameterName: str
    parameterValue: Optional[str] = None
    foundParameterValue: Optional[str] = None
    nearestHeading: Optional[str] = None
    isValueFound: bool = False
    pageNumber: Optional[int] = None
    lineNumber: Optional[int] = None
    matchingAccuracyPercentage: float = 0.0
    fileName: str

class ValueExtractionResponse(BaseModel):
    isSuccess: bool
    responseParameter: List[ExtractedParameter]
    extractionSummary: Optional[Dict[str, Any]] = None
    errorMessage: Optional[str] = None

class ParameterValueExtractor:
    def __init__(self, api_key: str):
        """Initialize the extractor with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Parameter type mappings for intelligent extraction
        self.parameter_mappings = {
            'vendor_name': ['vendor name', 'supplier', 'company name', 'from company', 'bill from', 'seller'],
            'invoice_number': ['invoice number', 'invoice no', 'inv no', 'invoice #', 'bill number', 'reference number'],
            'invoice_date': ['invoice date', 'date', 'bill date', 'issued date', 'created date'],
            'due_date': ['due date', 'payment due', 'due by', 'payable by'],
            'total_amount': ['total', 'amount due', 'total amount', 'balance due', 'grand total', 'final amount'],
            'tax_amount': ['tax', 'sales tax', 'vat', 'gst', 'tax amount'],
            'customer_name': ['bill to', 'customer', 'client', 'buyer', 'customer name'],
            'address': ['address', 'billing address', 'shipping address', 'location'],
            'phone': ['phone', 'telephone', 'contact', 'mobile', 'cell'],
            'email': ['email', 'e-mail', 'email address'],
            'description': ['description', 'item description', 'service description', 'details'],
            'quantity': ['qty', 'quantity', 'amount', 'count'],
            'unit_price': ['unit price', 'price', 'rate', 'cost per unit'],
            'subtotal': ['subtotal', 'sub total', 'net amount', 'before tax'],
            'payment_terms': ['payment terms', 'terms', 'payment conditions'],
            'po_number': ['po number', 'purchase order', 'po #', 'order number']
        }
    
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
    
    def preprocess_document(self, document: Document) -> Dict[str, Any]:
        """Preprocess document content to extract structure information"""
        text_content = self.decode_protobuf_content(document.fileContent)
        lines = text_content.split('\n')
        processed_lines = []
        headings = []
        
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            if stripped_line:
                # Enhanced heading detection for various document types
                if (stripped_line.isupper() and len(stripped_line) > 3) or \
                   re.match(r'^[A-Z][A-Z\s]{5,}$', stripped_line) or \
                   re.match(r'^\d+\.\s+[A-Z]', stripped_line) or \
                   any(keyword in stripped_line.upper() for keyword in [
                       'INVOICE', 'BILL', 'RECEIPT', 'VENDOR', 'CUSTOMER', 'BILLING', 
                       'SHIPPING', 'PAYMENT', 'TOTAL', 'SUMMARY', 'DETAILS', 'INFORMATION'
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
    
    def get_parameter_variants(self, parameter_name: str) -> List[str]:
        """Get possible variants/synonyms for a parameter name"""
        parameter_lower = parameter_name.lower()
        
        # Check if it matches any known parameter type
        for param_type, variants in self.parameter_mappings.items():
            if any(variant in parameter_lower for variant in variants):
                return variants
        
        # If not found in mappings, create variants based on the parameter name
        variants = [parameter_lower]
        
        # Add common variations
        if 'name' in parameter_lower:
            variants.extend([parameter_lower.replace('name', ''), parameter_lower.replace(' name', '')])
        if 'number' in parameter_lower:
            variants.extend([parameter_lower.replace('number', 'no'), parameter_lower.replace('number', '#')])
        if 'date' in parameter_lower:
            variants.append(parameter_lower.replace('date', ''))
        
        return list(set(variants))  # Remove duplicates
    
    def extract_value_with_ai(self, parameter_name: str, processed_doc: Dict) -> Dict[str, Any]:
        """Use AI to extract parameter value from document"""
        
        text_content = processed_doc['text_content']
        file_name = processed_doc['document_info']['fileName']
        
        # Get parameter variants for better matching
        parameter_variants = self.get_parameter_variants(parameter_name)
        
        prompt = f"""
        Analyze this document and extract the value for the parameter "{parameter_name}".
        
        Document content:
        {text_content[:8000]}  # Limit content to avoid token limits
        
        Parameter to find: "{parameter_name}"
        Possible variations: {parameter_variants}
        
        Instructions:
        1. Look for the parameter name or its variations in the document
        2. Extract the actual VALUE associated with that parameter
        3. For addresses, include the complete address (multiple lines if needed)
        4. For amounts, include currency symbols and proper formatting
        5. For dates, extract in the format found in the document
        6. For names/companies, extract the complete name as written
        
        Common patterns to look for:
        - "Parameter Name: Value"
        - "Parameter Name Value" (without colon)
        - Value appearing near parameter label
        - Value in tables or structured formats
        
        Return in this exact JSON format:
        {{
            "found": true/false,
            "extracted_value": "the actual value found or null",
            "context_line": "full line where the value was found or null",
            "parameter_location": "where the parameter name was found",
            "estimated_line_number": number or null,
            "confidence_score": percentage (0-100),
            "extraction_method": "how the value was extracted",
            "nearby_context": "surrounding text for context"
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
            logger.error(f"AI extraction error: {str(e)}")
            return {"found": False, "confidence_score": 0, "error": str(e)}
    
    def extract_value_direct(self, parameter_name: str, processed_doc: Dict) -> Dict[str, Any]:
        """Direct pattern-based value extraction"""
        lines = processed_doc['lines']
        headings = processed_doc['headings']
        parameter_variants = self.get_parameter_variants(parameter_name)
        
        best_match = {
            'found': False,
            'line_number': None,
            'extracted_value': None,
            'confidence': 0,
            'context_line': None,
            'nearest_heading': None,
            'extraction_method': 'direct_pattern'
        }
        
        # Pattern 1: Look for "Parameter: Value" patterns
        for line_data in lines:
            line_content = line_data['content']
            line_number = line_data['line_number']
            
            for variant in parameter_variants:
                # Pattern: "Parameter Name: Value"
                pattern1 = rf'{re.escape(variant)}\s*:\s*(.+?)(?:\s|$)'
                match1 = re.search(pattern1, line_content, re.IGNORECASE)
                
                if match1:
                    extracted_value = match1.group(1).strip()
                    confidence = 95
                    
                    best_match = {
                        'found': True,
                        'line_number': line_number,
                        'extracted_value': extracted_value,
                        'confidence': confidence,
                        'context_line': line_content,
                        'nearest_heading': self.find_nearest_heading(line_number, headings),
                        'extraction_method': 'colon_pattern'
                    }
                    return best_match
                
                # Pattern: Look for parameter name and extract value from next part of line
                if variant in line_content.lower():
                    # Try to extract value after the parameter name
                    param_index = line_content.lower().find(variant)
                    after_param = line_content[param_index + len(variant):].strip()
                    
                    # Remove common separators
                    for sep in [':', '-', '=', '|']:
                        after_param = after_param.lstrip(sep).strip()
                    
                    if after_param and len(after_param) > 0:
                        # Take the first meaningful part (until next field or end of line)
                        value_parts = after_param.split()
                        if value_parts:
                            extracted_value = ' '.join(value_parts[:5])  # Limit to reasonable length
                            confidence = 80
                            
                            if confidence > best_match['confidence']:
                                best_match = {
                                    'found': True,
                                    'line_number': line_number,
                                    'extracted_value': extracted_value,
                                    'confidence': confidence,
                                    'context_line': line_content,
                                    'nearest_heading': self.find_nearest_heading(line_number, headings),
                                    'extraction_method': 'inline_extraction'
                                }
        
        # Pattern 2: Look for structured data (tables, key-value pairs)
        if not best_match['found']:
            best_match = self.extract_from_structured_data(parameter_name, processed_doc)
        
        return best_match
    
    def extract_from_structured_data(self, parameter_name: str, processed_doc: Dict) -> Dict[str, Any]:
        """Extract values from structured data like tables"""
        lines = processed_doc['lines']
        parameter_variants = self.get_parameter_variants(parameter_name)
        
        # Look for table-like structures or field lists
        for i, line_data in enumerate(lines):
            line_content = line_data['content']
            
            for variant in parameter_variants:
                if variant in line_content.lower():
                    # Check if value is on the same line (separated by spaces/tabs)
                    parts = re.split(r'\s{2,}|\t', line_content)  # Split on multiple spaces or tabs
                    
                    for j, part in enumerate(parts):
                        if variant in part.lower() and j + 1 < len(parts):
                            extracted_value = parts[j + 1].strip()
                            if extracted_value:
                                return {
                                    'found': True,
                                    'line_number': line_data['line_number'],
                                    'extracted_value': extracted_value,
                                    'confidence': 85,
                                    'context_line': line_content,
                                    'nearest_heading': self.find_nearest_heading(line_data['line_number'], processed_doc['headings']),
                                    'extraction_method': 'structured_data'
                                }
                    
                    # Check if value is on the next line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]['content'].strip()
                        if next_line and len(next_line) < 100:  # Reasonable value length
                            return {
                                'found': True,
                                'line_number': lines[i + 1]['line_number'],
                                'extracted_value': next_line,
                                'confidence': 75,
                                'context_line': f"{line_content} -> {next_line}",
                                'nearest_heading': self.find_nearest_heading(lines[i + 1]['line_number'], processed_doc['headings']),
                                'extraction_method': 'next_line'
                            }
        
        return {'found': False, 'confidence': 0}
    
    def post_process_extracted_value(self, parameter_name: str, extracted_value: str) -> str:
        """Post-process extracted value based on parameter type"""
        if not extracted_value:
            return extracted_value
        
        parameter_lower = parameter_name.lower()
        
        # Clean up common artifacts
        cleaned_value = extracted_value.strip()
        
        # Remove common trailing punctuation
        cleaned_value = re.sub(r'[,;\.]+$', '', cleaned_value)
        
        # Date formatting
        if 'date' in parameter_lower:
            # Try to standardize date format while preserving original
            date_match = re.search(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', cleaned_value)
            if date_match:
                return date_match.group(0)
        
        # Amount formatting
        if any(word in parameter_lower for word in ['amount', 'total', 'price', 'cost']):
            # Extract monetary values
            amount_match = re.search(r'\$?[\d,]+\.?\d*', cleaned_value)
            if amount_match:
                return amount_match.group(0)
        
        # Email extraction
        if 'email' in parameter_lower:
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', cleaned_value)
            if email_match:
                return email_match.group(0)
        
        # Phone number extraction
        if 'phone' in parameter_lower or 'telephone' in parameter_lower:
            phone_match = re.search(r'[\(\d]{1}[\d\-\.\s\(\)]{7,15}', cleaned_value)
            if phone_match:
                return phone_match.group(0).strip()
        
        return cleaned_value
    
    def extract_parameter_value(self, parameter_name: str, processed_doc: Dict) -> ExtractedParameter:
        """Extract value for a single parameter from the document"""
        
        # Try direct extraction first
        direct_result = self.extract_value_direct(parameter_name, processed_doc)
        
        # Use AI extraction if direct method failed or has low confidence
        if not direct_result['found'] or direct_result['confidence'] < 70:
            ai_result = self.extract_value_with_ai(parameter_name, processed_doc)
            
            # Choose the better result
            if ai_result.get('found') and ai_result.get('confidence_score', 0) > direct_result['confidence']:
                extracted_value = ai_result.get('extracted_value')
                confidence = ai_result.get('confidence_score', 0)
                line_number = ai_result.get('estimated_line_number')
                context_line = ai_result.get('context_line')
            else:
                extracted_value = direct_result.get('extracted_value')
                confidence = direct_result.get('confidence', 0)
                line_number = direct_result.get('line_number')
                context_line = direct_result.get('context_line')
        else:
            extracted_value = direct_result.get('extracted_value')
            confidence = direct_result.get('confidence', 0)
            line_number = direct_result.get('line_number')
            context_line = direct_result.get('context_line')
        
        # Post-process the extracted value
        if extracted_value:
            extracted_value = self.post_process_extracted_value(parameter_name, extracted_value)
        
        # Find nearest heading if we have a line number
        nearest_heading = None
        if line_number:
            nearest_heading = self.find_nearest_heading(line_number, processed_doc['headings'])
        
        return ExtractedParameter(
            parameterName=parameter_name,
            parameterValue=extracted_value,
            foundParameterValue=extracted_value,  # Same as parameterValue in extraction mode
            nearestHeading=nearest_heading,
            isValueFound=bool(extracted_value and confidence > 0),
            pageNumber=1,  # Assuming single page for text documents
            lineNumber=line_number,
            matchingAccuracyPercentage=float(confidence),
            fileName=processed_doc['document_info']['fileName']
        )
    
    def generate_extraction_summary(self, results: List[ExtractedParameter]) -> Dict[str, Any]:
        """Generate summary of extraction results"""
        total_parameters = len(results)
        found_parameters = sum(1 for r in results if r.isValueFound)
        
        summary = {
            'total_parameters_requested': total_parameters,
            'parameters_found': found_parameters,
            'extraction_success_rate': round((found_parameters / total_parameters) * 100, 1) if total_parameters > 0 else 0,
            'average_confidence': round(sum(r.matchingAccuracyPercentage for r in results) / total_parameters, 1) if total_parameters > 0 else 0,
            'parameters_by_confidence': {
                'high_confidence': [r.parameterName for r in results if r.matchingAccuracyPercentage >= 90],
                'medium_confidence': [r.parameterName for r in results if 70 <= r.matchingAccuracyPercentage < 90],
                'low_confidence': [r.parameterName for r in results if 0 < r.matchingAccuracyPercentage < 70],
                'not_found': [r.parameterName for r in results if not r.isValueFound]
            },
            'extraction_methods_used': list(set([
                'ai_extraction' if r.matchingAccuracyPercentage > 0 else 'not_found' 
                for r in results
            ]))
        }
        
        return summary
    
    def extract_values(self, request: ValueExtractionRequest) -> ValueExtractionResponse:
        """Main method to extract parameter values from document"""
        try:
            # Preprocess the document
            logger.info(f"Processing document: {request.document.fileName}")
            processed_doc = self.preprocess_document(request.document)
            
            # Extract values for each parameter
            extracted_parameters = []
            
            for param_name in request.parameterNames:
                logger.info(f"Extracting value for parameter: {param_name}")
                
                result = self.extract_parameter_value(param_name, processed_doc)
                extracted_parameters.append(result)
            
            # Generate extraction summary
            extraction_summary = self.generate_extraction_summary(extracted_parameters)
            
            return ValueExtractionResponse(
                isSuccess=True,
                responseParameter=extracted_parameters,
                extractionSummary=extraction_summary
            )
            
        except Exception as e:
            logger.error(f"Error extracting values: {str(e)}")
            return ValueExtractionResponse(
                isSuccess=False,
                responseParameter=[],
                errorMessage=str(e)
            )

# Global extractor instance
extractor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global extractor
    
    # Startup
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set!")
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    extractor = ParameterValueExtractor(api_key)
    logger.info("ParameterValueExtractor initialized successfully")
    
    yield  # This is where the app runs
    
    # Shutdown (if needed)
    logger.info("Application shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Parameter Value Extraction API",
    description="API to extract parameter values from documents using AI",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/api/extract-values", response_model=ValueExtractionResponse)
async def extract_values(request: ValueExtractionRequest):
    """
    Extract parameter values from a document
    
    - **document**: Document to extract values from (protobuf format, base64 encoded)
    - **parameterNames**: List of parameter names to find values for
    
    Returns extracted values with confidence scores and location information.
    """
    if extractor is None:
        raise HTTPException(status_code=500, detail="Service not initialized properly")
    
    try:
        result = extractor.extract_values(request)
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
        "message": "Parameter Value Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "extract_values": "/api/extract-values",
            "health": "/health",
            "docs": "/docs"
        },
        "supported_parameters": [
            "Vendor Name", "Invoice Number", "Invoice Date", "Due Date",
            "Total Amount", "Tax Amount", "Customer Name", "Address",
            "Phone", "Email", "Description", "Quantity", "Unit Price"
        ]
    }

# Example usage and testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)