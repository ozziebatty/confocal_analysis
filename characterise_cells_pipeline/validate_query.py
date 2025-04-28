import pandas as pd

def validate_image_format():
    #Load image
    #Extract shape

def validate_parameters():

    
    def check_parameters(file_path):
        # Load the Excel file
        df = pd.read_excel(file_path)

        # List to collect all validation errors
        errors = []

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            parameter = row['Parameter']
            section = row['Section']
            value = row['Value']
            default_value = row['Default Value']
            min_val = row['Min']
            max_val = row['Max']
            data_type = row['Data type']
            must_be_odd = row['Must be odd']

            # Use default value if 'Value' is not present
            if pd.isna(value):
                value = default_value

            # Check data type
            try:
                if data_type == 'Integer':
                    float_value = float(value)
                    if not float_value.is_integer():
                        raise ValueError(f"Value {value} is not a valid integer for parameter {parameter}.")
                    value = int(value)
                elif data_type == 'Float':
                    value = float(value)
                else:
                    errors.append(f"Invalid data type for parameter {parameter}. Expected 'Integer' or 'Float'.")
                    continue  # Skip further checks for this parameter
            except ValueError:
                errors.append(f"Data type mismatch for parameter {parameter}. Expected {data_type}.")
                continue  # Skip further checks for this parameter

            # Check if value is within min and max range
            if not (min_val <= value <= max_val):
                errors.append(f"Value for parameter {parameter} is out of range. Expected between {min_val} and {max_val}, got {value}.")

            # Check if value must be odd
            if must_be_odd and data_type == 'Integer':
                if value % 2 == 0:
                    errors.append(f"Value for parameter {parameter} must be odd, got {value}.")

        # Raise all collected errors at the end
        if errors:
            raise ValueError("\n".join(errors))

    # Path to the Excel file
    file_path = "/Users/oskar/Desktop/steventon_lab/image_analysis/imaging_data/parameters.xlsx"

    try:
        check_parameters(file_path)
        print("All parameters are valid.")
    except ValueError as e:
        print(e)

validate_image_format()
validate_parameters()

print("Run succesfully")