#!/usr/bin/env python3
"""
Test script to demonstrate dynamic question generation
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Import the generators from our app
from app import dynamic_generator, template_generator

def test_template_generation():
    """Test template-based question generation"""
    print("=" * 60)
    print("TEMPLATE-BASED QUESTION GENERATION")
    print("=" * 60)
    
    subjects = ['Python', 'Java']
    difficulties = ['Easy', 'Medium', 'Hard']
    
    for subject in subjects:
        print(f"\n📚 Subject: {subject}")
        print("-" * 40)
        
        for difficulty in difficulties:
            print(f"\n🎯 Difficulty: {difficulty}")
            questions = template_generator.generate_questions(subject, difficulty, 3)
            
            for i, question in enumerate(questions, 1):
                print(f"  {i}. {question}")

def test_dynamic_generation():
    """Test OpenAI-based dynamic question generation"""
    print("\n" + "=" * 60)
    print("DYNAMIC QUESTION GENERATION (OpenAI API)")
    print("=" * 60)
    
    if not dynamic_generator:
        print("❌ OpenAI Dynamic Generator not available (no API key)")
        return
    
    subjects = ['Python', 'Machine Learning']
    difficulties = ['Easy', 'Hard']
    
    for subject in subjects:
        print(f"\n📚 Subject: {subject}")
        print("-" * 40)
        
        for difficulty in difficulties:
            print(f"\n🎯 Difficulty: {difficulty}")
            try:
                questions = dynamic_generator.generate_questions(subject, difficulty, 3)
                
                for i, question in enumerate(questions, 1):
                    print(f"  {i}. {question}")
            except Exception as e:
                print(f"  ❌ Error: {e}")

def main():
    """Main test function"""
    print("🚀 Testing AI Question Generation System")
    print(f"OpenAI API Key configured: {'✅' if os.environ.get('OPENAI_API_KEY') else '❌'}")
    
    # Test template generation (always works)
    test_template_generation()
    
    # Test dynamic generation (requires OpenAI API key)
    test_dynamic_generation()
    
    print("\n" + "=" * 60)
    print("✅ Testing Complete!")
    print("To use OpenAI dynamic generation, add your API key to .env file:")
    print("OPENAI_API_KEY=your-actual-openai-api-key-here")
    print("=" * 60)

if __name__ == "__main__":
    main()
