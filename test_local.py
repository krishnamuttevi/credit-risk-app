#!/usr/bin/env python3
"""
Local testing script for Credit Risk Prediction API
Run this before deploying to ensure everything works correctly
"""

import requests
import json
import sys
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if the API is running"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on port 8000")
        return False

def test_home_page():
    """Test if the home page loads"""
    print("\nTesting home page...")
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            if "Credit Risk Assessment" in response.text:
                print("✅ Home page loads correctly")
                return True
            else:
                print("❌ Home page loads but content is unexpected")
                return False
        else:
            print(f"❌ Home page failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error loading home page: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\nTesting single prediction...")
    
    # Sample data for testing
    test_data = {
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 150000,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 450000,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -12775,
        "DAYS_EMPLOYED": -2003,
        "DAYS_REGISTRATION": -4000,
        "DAYS_ID_PUBLISH": -2000,
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": 1,
        "FLAG_WORK_PHONE": 0,
        "FLAG_CONT_MOBILE": 1,
        "FLAG_PHONE": 0,
        "FLAG_EMAIL": 1,
        "REGION_POPULATION_RELATIVE": 0.018,
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
        "HOUR_APPR_PROCESS_START": 10
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"   Loan Approved: {result.get('loan_approved')}")
            print(f"   Default Probability: {result.get('default_probability', 0)*100:.1f}%")
            print(f"   Risk Level: {result.get('risk_level')}")
            print(f"   Confidence Score: {result.get('confidence_score', 0)*100:.1f}%")
            return True
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return False

def test_api_docs():
    """Test if API documentation is accessible"""
    print("\nTesting API documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API documentation is accessible")
            print(f"   Visit {BASE_URL}/docs to see interactive docs")
            return True
        else:
            print(f"❌ API docs failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing API docs: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            info = response.json()
            print("✅ Model info retrieved")
            print(f"   Model Type: {info.get('model_type', 'Unknown')}")
            print(f"   Model Loaded: {info.get('model_loaded', False)}")
            print(f"   Number of Features: {info.get('n_features', 'Unknown')}")
            return True
        else:
            print(f"❌ Model info failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Credit Risk Prediction API - Local Testing")
    print("=" * 50)
    
    # Check if server is running
    print("\n⚠️  Make sure the server is running with: python main.py")
    print("Waiting for server to be ready...\n")
    time.sleep(2)
    
    # Run tests
    tests_passed = 0
    total_tests = 5
    
    if test_health_check():
        tests_passed += 1
    
    if test_home_page():
        tests_passed += 1
    
    if test_api_docs():
        tests_passed += 1
    
    if test_model_info():
        tests_passed += 1
    
    if test_single_prediction():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Summary: {tests_passed}/{total_tests} tests passed")
    print("=" * 50)
    
    if tests_passed == total_tests:
        print("\n✅ All tests passed! Your API is ready for deployment.")
        return 0
    else:
        print(f"\n❌ {total_tests - tests_passed} tests failed. Please fix the issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())