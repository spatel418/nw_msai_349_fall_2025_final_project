import pandera as pa
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get the root logger
logger = logging.getLogger()  # Or logging.getLogger(__name__) for module-specific

class RegressionFileValidator:
    def __init__(self, headers):
        # Create a dictionary of validation schemas for each column
        validation_dict = {}
        # Loop through each header and create a validation for it
        for header in headers:
            validation_dict[header] = pa.Column(
                pa.Object,
                [
                    pa.Check(
                        lambda s: s.notnull(),
                        error=f"{header} should not have null rows.",
                    ),
                    pa.Check(
                        lambda s: s.apply(lambda x: isinstance(x, (int, float))),
                        error=f"{header} should be an integer or float value.",
                    ),
                ],
            )

        # Create the validation schema with each column's checks
        self.validation_schema = pa.DataFrameSchema(validation_dict)

    def run_validation(self, df):
        try:
            df = df.astype(object)
            df = self.validation_schema.validate(df)
            return df
        except pa.errors.SchemaError as e:
            # removing this text as this error is returned to the front end and this makes the message cleaner.
            return f"{e}".replace(": <Check <lambda>", "")
