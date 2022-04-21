import os
import git
import pandas as pd

repo = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")


def create_overall_dataframes(path):
    directory = os.fsencode(path)
    df_rides = pd.DataFrame()
    df_kpi = pd.DataFrame()
    df_mtd = pd.DataFrame()
    for filename in os.listdir(directory):
        # Get correct filename
        current_file = os.path.join(directory, filename).decode("utf-8")
        current_file = current_file.replace("$", "")
        current_file = current_file.replace("~", "")

        # Read current excel file
        df_dict = pd.read_excel(current_file, sheet_name=None)

        for sheet_name, df in df_dict.items():
            df["sheet_name"] = sheet_name

        # Extract dataframes from current excel file
        df_kpi_temp = df_dict["KPI"]
        df_dict.pop("KPI")
        df_mtd_temp = df_dict["MTD"]
        df_dict.pop("MTD")
        # union all rides from all days in current excel file
        df_rides_temp = pd.concat(df_dict, ignore_index=True)
        df_rides_temp["file_name"] = str(filename)

        # Create big dataframes over all excel files (all months combined)
        df_kpi = pd.concat([df_kpi, df_kpi_temp], axis=0, ignore_index=True)
        df_mtd = pd.concat([df_mtd, df_mtd_temp], axis=0, ignore_index=True)
        df_rides = pd.concat([df_rides, df_rides_temp], axis=0, ignore_index=True)

    return {"df_kpi": df_kpi, "df_mtd": df_mtd, "df_rides": df_rides}


result = create_overall_dataframes(f"{repo}/data/normal_rides")

result["df_kpi"].to_csv(f"{repo}/data/kpi_combined.csv")
result["df_mtd"].to_csv(f"{repo}/data/mtd_combined.csv")
result["df_rides"].to_csv(f"{repo}/data/rides_combined.csv")