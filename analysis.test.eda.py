#unit test file
import pytest
import pandas as pd
from analysis import load_data, inspect_data, get_unique_cities

@pytest.fixture
def sample_df():
    """Provides a small dummy dataframe for fast and reliable testing."""
    return pd.DataFrame({
        'city': ['Los Angeles', 'New York', 'Los Angeles'],
        'country': ['USA', 'USA', 'USA'],
        'date': ['2019-01-01', '2019-01-01', '2019-02-01'],
        'pm25': [91.0, 50.0, 80.0],
        'pm10': [40.0, 30.0, 35.0],
        'no2': [20.0, 15.0, 18.0],
        'so2': [5.0, 4.0, 4.5],
        'co': [1.2, 0.9, 1.0],
        'o3': [30.0, 25.0, 28.0],
        'temperature_c': [22.0, 18.0, 20.0],
        'humidity': [51.0, 60.0, 55.0],
        'wind_kph': [9.8, 11.2, 10.0],
        'aqi': [93.0, 55.0, 82.0],
        'category': ['Moderate', 'Good', 'Moderate']
    })

def test_load_data():
    """Test that load_data returns a non-empty pandas DataFrame."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 0

def test_get_unique_cities(sample_df):
    """Test that unique cities are extracted properly."""
    cities = sample_df['city'].unique()
    assert len(cities) == 2
    assert 'Los Angeles' in cities
    assert 'New York' in cities