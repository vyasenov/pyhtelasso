#!/usr/bin/env python3
"""
Simple test script for pyhtelasso package

This script tests basic functionality to ensure the package works correctly.
"""

import sys
import numpy as np

def test_import():
    """Test that the package can be imported"""
    try:
        from pyhtelasso import HTELasso, simulate_data, ECONML_AVAILABLE
        print("✓ Package imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_simulation():
    """Test data simulation"""
    try:
        from pyhtelasso import simulate_data
        
        # Test basic simulation
        X, y, t, true_mod = simulate_data(
            n=100, p_features=10, p_het=2, 
            treatment_prob=0.5, random_state=42
        )
        
        # Check shapes
        assert X.shape == (100, 10), f"Expected X shape (100, 10), got {X.shape}"
        assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
        assert t.shape == (100,), f"Expected t shape (100,), got {t.shape}"
        assert len(true_mod) == 2, f"Expected 2 true moderators, got {len(true_mod)}"
        
        print("✓ Data simulation works correctly")
        return True
    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        return False

def test_detector():
    """Test HTELasso"""
    try:
        from pyhtelasso import HTELasso, simulate_data
        
        # Generate test data
        X, y, t, true_mod = simulate_data(
            n=200, p_features=15, p_het=3, 
            treatment_prob=0.4, random_state=42
        )
        
        # Initialize detector
        detector = HTELasso(
            n_bootstrap=100,  # Small number for quick test
            bootstrap_alpha=0.05,
            random_state=42
        )
        
        # Fit model
        detector.fit(X, y, t, p=t.mean())
        
        # Test prediction
        het_scores = detector.predict_heterogeneity(X[:5])
        assert len(het_scores) == 5, f"Expected 5 predictions, got {len(het_scores)}"
        
        # Test results
        feature_names = [f"X{i}" for i in range(X.shape[1])]
        results = detector.get_significant_variables(feature_names)
        
        assert 'significant_indices' in results, "Missing significant_indices in results"
        assert 'significant_features' in results, "Missing significant_features in results"
        assert 'coefficients' in results, "Missing coefficients in results"
        
        print("✓ HTELasso works correctly")
        return True
    except Exception as e:
        print(f"✗ Detector test failed: {e}")
        return False

def test_summary_table():
    """Test summary table generation"""
    try:
        from pyhtelasso import HTELasso, simulate_data
        
        # Generate test data
        X, y, t, true_mod = simulate_data(
            n=200, p_features=10, p_het=2, 
            treatment_prob=0.4, random_state=42
        )
        
        # Fit detector
        detector = HTELasso(
            n_bootstrap=100,
            bootstrap_alpha=0.05,
            random_state=42
        )
        detector.fit(X, y, t, p=t.mean())
        
        # Generate summary table
        feature_names = [f"X{i}" for i in range(X.shape[1])]
        summary_df = detector.summary_table(feature_names)
        
        # Check table structure
        assert 'feature' in summary_df.columns, "Missing 'feature' column"
        assert 'coefficient' in summary_df.columns, "Missing 'coefficient' column"
        assert 'significant' in summary_df.columns, "Missing 'significant' column"
        assert len(summary_df) == X.shape[1], f"Expected {X.shape[1]} rows, got {len(summary_df)}"
        
        print("✓ Summary table generation works correctly")
        return True
    except Exception as e:
        print(f"✗ Summary table test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing pyhtelasso package...")
    print("="*50)
    
    tests = [
        ("Import", test_import),
        ("Simulation", test_simulation),
        ("Detector", test_detector),
        ("Summary Table", test_summary_table)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} test failed")
    
    print("\n" + "="*50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Package is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 