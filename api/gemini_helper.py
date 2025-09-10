import logging
import requests
import json
import os
from typing import Dict, Optional, Tuple
import re
from dotenv import load_dotenv # Import load_dotenvs

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# OpenRouter API configuration as global variables
API_KEY = os.getenv('OPENROUTER_API_KEY')
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash"  # Gemini 2.5 Flash model
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:5000",
    "X-Title": "Farm Voice Assistant"
}

if not API_KEY:
    logger.error("OPENROUTER_API_KEY environment variable not set!")
else:
    logger.info("Gemini Agriculture Assistant initialized")

def create_agriculture_prompt(bemba_text: str) -> str:
    """
    Create a comprehensive prompt for agricultural assistance in Bemba
    """
    return f"""
You are an expert agricultural assistant specializing in Zambian farming practices. 

The user has spoken in Bemba language: "{bemba_text}"

Your task is to:
1. Understand the agricultural question or request in Bemba
2. Provide detailed, practical farming advice
3. Respond ONLY in Bemba language
4. Focus on crops commonly grown in Zambia (maize/amataba, beans/chilemba, groundnuts/imbalala, sweet potatoes/ifumbu, cassava/tute/kalundwe, vegetables/umusalu)

If the question is about:
- **Maize cultivation (amataba)**: Provide step-by-step planting, fertilizing, pest control, and harvesting advice
- **Crop diseases**: Explain symptoms and organic/affordable treatments
- **Soil preparation**: Describe land preparation, composting, and soil health
- **Planting seasons**: Explain best times for different crops in Zambia
- **Pest control**: Suggest natural and affordable pest management
- **Harvesting**: Explain when and how to harvest different crops
- **Storage**: Provide advice on proper crop storage to prevent spoilage

Important guidelines:
- Use simple, clear Bemba that rural farmers can understand
- Focus on affordable, accessible farming methods
- Include traditional Zambian farming knowledge
- Provide step-by-step instructions when possible
- Mention specific timeframes (days, weeks, months)
- Consider Zambian climate and seasons

Respond with practical, actionable advice in proper Bemba language only.
"""

def translate_and_get_agriculture_advice(bemba_text: str) -> Dict:
    """
    Send Bemba text to Gemini for agricultural advice and translation
    """
    if not API_KEY:
        logger.error("OpenRouter API key not configured")
        return {
            'success': False,
            'error': 'API key not configured',
            'bemba_response': 'Tapali ukufunda kwa API. Mwasuma administrator.'
        }

    try:
        # Create the agricultural prompt
        prompt = create_agriculture_prompt(bemba_text)
        
        # Prepare the request payload
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert agricultural assistant for Zambian farmers. Always respond in Bemba language with practical farming advice."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }

        logger.info(f"Sending request to Gemini for: '{bemba_text[:50]}...'")
        
        # Make the API request
        response = requests.post(
            BASE_URL,
            headers=HEADERS,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                bemba_advice = result['choices'][0]['message']['content'].strip()
                
                logger.info(f"Gemini response received: '{bemba_advice[:100]}...'")
                
                return {
                    'success': True,
                    'original_bemba': bemba_text,
                    'bemba_response': bemba_advice,
                    'model_used': MODEL,
                    'tokens_used': result.get('usage', {}).get('total_tokens', 0)
                }
            else:
                logger.error(f"Unexpected response format: {result}")
                return {
                    'success': False,
                    'error': 'Invalid response format from Gemini',
                    'bemba_response': 'Ichipepesho cha API tachikonkele bwino. Yesubeni nangu.'
                }
        else:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            return {
                'success': False,
                'error': f'API request failed: {response.status_code}',
                'bemba_response': 'Ukutumina kwa API kwalekelele. Yesubeni nangu.'
            }

    except requests.exceptions.Timeout:
        logger.error("Request timeout to Gemini API")
        return {
            'success': False,
            'error': 'Request timeout',
            'bemba_response': 'Icitali chalekelele. Yesubeni nangu.'
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        return {
            'success': False,
            'error': f'Request failed: {str(e)}',
            'bemba_response': 'Ukutumina kwalekelele. Yendani mukafunde intaneti.'
        }
    except Exception as e:
        logger.error(f"Unexpected error in Gemini request: {e}")
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}',
            'bemba_response': 'Ichipepesho chalekelele. Yesubeni nangu.'
        }

def is_agriculture_related(text: str) -> bool:
    """
    Check if the text is related to agriculture/farming
    """
    agriculture_keywords_bemba = [
        'kandolo', 'maize', 'bushe', 'farming', 'nyemba', 'beans',
        'tute', 'groundnuts', 'ukubomfya', 'sweet potato', 'manioka',
        'cassava', 'kubiyala', 'planting', 'ukufyeka', 'harvesting',
        'ifisama', 'seeds', 'umutaba', 'fertilizer', 'amataba',
        'mushani', 'garden', 'mulimi', 'farmer', 'abalimi', 'farmers',
        'ifya kulima', 'farming things', 'ukupalula', 'weeding',
        'ukudiila', 'irrigation', 'ifipushi', 'insects', 'ukukolola',
        'harvesting', 'ukubika', 'kalundwele', 'storage'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in agriculture_keywords_bemba)

def get_agriculture_advice(bemba_text: str) -> Dict:
    """
    Main function to get agricultural advice for Bemba text
    """
    logger.info(f"Getting agriculture advice for: '{bemba_text[:50]}...'")
    
    # Check if the query is agriculture-related
    # if not is_agriculture_related(bemba_text):
    #     logger.info("Query not agriculture-related, providing general response")
    #     return {
    #         'success': True,
    #         'original_bemba': bemba_text,
    #         'bemba_response': 'Ninshi muleipusha ipusheni. ifipusho fyapa bulimi , nangu ifya kubiyala.',
    #         'model_used': 'local_filter',
    #         'tokens_used': 0,
    #         'is_agriculture': False
    #     }
    
    # Get advice from Gemini
    result = translate_and_get_agriculture_advice(bemba_text)
    result['is_agriculture'] = True
    
    return result

def test_gemini_connection() -> bool:
    """
    Test if Gemini API connection is working
    """
    try:
        test_result = get_agriculture_advice("bushe nalimisha bwanji chimanga?")
        return test_result['success']
    except Exception as e:
        logger.error(f"Gemini connection test failed: {e}")
        return False
