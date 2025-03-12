from io import StringIO
import requests
import pandas as pd

# Import sensitive configuration from config.py
try:
    from config import API_KEY, EMAIL
except ImportError:
    print("Warning: config.py not found. Please create it with your API_KEY and EMAIL.")
    print("Example config.py content:")
    print("API_KEY = 'your_nrel_api_key'")
    print("EMAIL = 'your_email@example.com'")
    exit(1)

# Direct CSV Download
WKT_POINT = "POINT(-115.3939 33.8214)"  # Desert Sunlight Solar Farm Center
URL_CSV = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"

# Request parameters (base parameters without year)
PARAMS = {
    "api_key": API_KEY,
    "wkt": WKT_POINT,
    "attributes": "air_temperature,ghi,clearsky_ghi",
    "utc": "true",
    "leap_day": "false",
    "interval": "60",  # Hourly data
    "email": EMAIL,
}


def fetch_and_filter_csv(year):
    """Fetches CSV data, filters for summer clear days, and saves the result for a specific year."""
    # add the year to the parameters
    params = PARAMS.copy()
    params["names"] = str(year)
    
    response = requests.get(URL_CSV, params=params)

    if response.status_code == 200:
        # load CSV content into a Pandas dataframe
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, skiprows=2)  # Skip metadata rows

        # print actual column names retrieved
        print(f"Column Names in Retrieved CSV for {year}:", df.columns.tolist())

        # standardize column names (strip spaces, convert to lowercase)
        df.columns = df.columns.str.strip().str.lower()

        # column name mapping (API returned different names)
        column_mapping = {
            "temperature": "air_temperature",
            "ghi": "ghi",
            "clearsky ghi": "clearsky_ghi",
        }

        # rename columns based on mapping
        df = df.rename(columns=column_mapping)

        # Convert time components to datetime format
        df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

        # Filter for summer months (June, July, August, September)
        df = df[df["month"].isin([6, 7, 8, 9])]

        # Filter for daytime hours (only when GHI > 50 W/m²)
        df = df[df["ghi"] > 50]

        # Filter for clear days (when GHI is within 5% of clearsky GHI)
        df = df[abs(df["ghi"] - df["clearsky_ghi"]) < 0.05 * df["clearsky_ghi"]]

        # Select relevant columns
        df = df[["time", "air_temperature", "ghi"]]

        # Save filtered dataset to CSV for this year
        output_file = f"mojave_summer_clear_days_{year}.csv"
        df.to_csv(output_file, index=False)
        print(f"Filtered data for {year} saved to {output_file} with {len(df)} rows")
        
        return df


def merge_csv_files(years=[2020, 2021, 2022, 2023]):
    """Merge the CSV files from different years into a single file."""

    # fetch and filter data for each year
    dataframes = []
    for year in years:
        df = fetch_and_filter_csv(year)
        if df is not None:
            dataframes.append(df)
            print(f"Added data for {year} with {len(df)} rows")
    
    # combine the dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined data from {len(dataframes)} years: {len(combined_df)} rows")
    
    # save the combined dataset
    combined_df.to_csv("mojave_summer_clear_days.csv", index=False)
    print(f"Combined data saved to mojave_summer_clear_days.csv with {len(combined_df)} rows")


# run the function to merge data from specified years
merge_csv_files()
