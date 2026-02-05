"""
Test for Data Loader Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_loader import DataLoader


@pytest.fixture
def sample_dataframe():
    """Crea un DataFrame di esempio per i test"""
    data = {
        'ID': [1, 2, 3, 4, 5],
        'CODE_GENDER': ['M', 'F', 'M', 'F', 'M'],
        'FLAG_OWN_CAR': ['Y', 'N', 'Y', 'Y', 'N'],
        'FLAG_OWN_REALTY': ['Y', 'Y', 'N', 'Y', 'N'],
        'CNT_CHILDREN': [0, 1, 2, 0, 1],
        'AMT_INCOME_TOTAL': [50000, 60000, 45000, 70000, 55000],
        'NAME_INCOME_TYPE': ['Working', 'Working', 'Commercial', 'Working', 'Working'],
        'NAME_EDUCATION_TYPE': ['Higher', 'Secondary', 'Higher', 'Higher', 'Secondary'],
        'NAME_FAMILY_STATUS': ['Married', 'Single', 'Married', 'Married', 'Single'],
        'NAME_HOUSING_TYPE': ['House', 'Apartment', 'House', 'House', 'Apartment'],
        'DAYS_BIRTH': [-10000, -12000, -15000, -11000, -9000],
        'DAYS_EMPLOYED': [-2000, -3000, 100, -4000, -1500],
        'FLAG_MOBIL': [1, 1, 1, 1, 1],
        'FLAG_WORK_PHONE': [1, 0, 1, 1, 0],
        'FLAG_PHONE': [0, 1, 0, 1, 1],
        'FLAG_EMAIL': [1, 1, 0, 1, 1],
        'OCCUPATION_TYPE': ['Engineer', 'Teacher', 'Manager', 'Engineer', 'Teacher'],
        'CNT_FAM_MEMBERS': [2, 1, 4, 2, 1],
        'TARGET': [1, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(tmp_path, sample_dataframe):
    """Crea un file CSV temporaneo per i test"""
    file_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return str(file_path)


class TestDataLoader:
    """Test suite per DataLoader"""
    
    def test_init(self):
        """Test inizializzazione DataLoader"""
        loader = DataLoader()
        assert loader.df is None
        assert loader.data_path is None
    
    def test_init_with_path(self, temp_csv_file):
        """Test inizializzazione con path"""
        loader = DataLoader(temp_csv_file)
        assert loader.data_path == temp_csv_file
    
    def test_load_data_success(self, temp_csv_file):
        """Test caricamento dati con successo"""
        loader = DataLoader()
        df = loader.load_data(temp_csv_file)
        
        assert df is not None
        assert len(df) == 5
        assert 'TARGET' in df.columns
    
    def test_load_data_file_not_found(self):
        """Test caricamento con file inesistente"""
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_data("non_existent_file.csv")
    
    def test_validate_data(self, temp_csv_file):
        """Test validazione dati"""
        loader = DataLoader()
        df = loader.load_data(temp_csv_file)
        
        # Dovrebbe completare senza errori
        loader._validate_data()
    
    def test_get_data_info(self, temp_csv_file):
        """Test ottenimento informazioni dataset"""
        loader = DataLoader()
        loader.load_data(temp_csv_file)
        
        info = loader.get_data_info()
        
        assert info['n_rows'] == 5
        assert info['n_columns'] == 19
        assert 'TARGET' in info['columns']
        assert isinstance(info['duplicates'], (int, np.integer))
    
    def test_check_target_distribution(self, temp_csv_file):
        """Test distribuzione target"""
        loader = DataLoader()
        loader.load_data(temp_csv_file)
        
        dist = loader.check_target_distribution()
        
        assert 'counts' in dist
        assert 'proportions' in dist
        assert 'class_ratio' in dist
        assert 0 in dist['counts']
        assert 1 in dist['counts']
    
    def test_check_target_distribution_missing_target(self, temp_csv_file):
        """Test distribuzione target con colonna mancante"""
        loader = DataLoader()
        df = loader.load_data(temp_csv_file)
        loader.df = df.drop(columns=['TARGET'])
        
        with pytest.raises(ValueError):
            loader.check_target_distribution()


class TestDataLoaderEdgeCases:
    """Test per casi limite"""
    
    def test_empty_dataframe(self, tmp_path):
        """Test con DataFrame vuoto"""
        empty_df = pd.DataFrame()
        file_path = tmp_path / "empty.csv"
        empty_df.to_csv(file_path, index=False)
        
        loader = DataLoader()
        df = loader.load_data(str(file_path))
        
        assert len(df) == 0
    
    def test_missing_required_columns(self, tmp_path):
        """Test con colonne mancanti"""
        df = pd.DataFrame({
            'ID': [1, 2],
            'TARGET': [0, 1]
        })
        
        file_path = tmp_path / "incomplete.csv"
        df.to_csv(file_path, index=False)
        
        loader = DataLoader()
        # Dovrebbe caricare ma mostrare warning
        loaded_df = loader.load_data(str(file_path))
        
        assert loaded_df is not None
        assert len(loaded_df) == 2


if __name__ == "__main__":
    # Esegui test
    pytest.main([__file__, "-v"])
