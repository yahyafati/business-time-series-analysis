from main_app import app


def main():
    file_path = "data.csv"
    n = 2  # number of non-time series columns
    final_df = app.run(file_path, n)
    final_df.to_csv("output.ignore.csv", index=True)


if __name__ == "__main__":
    main()
