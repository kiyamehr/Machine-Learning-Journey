def clean_titanic_data(dataset):
    """
    A Function used for cleaning the dataset \
        from the Titanic surivival prediction Challenge
    """
    # Removing unrealated Columns
    cleaned_data = dataset.drop(columns=["Name", "PassengerId"])

    # Changing genders to numbers
    cleaned_data = cleaned_data.rename(columns={"Sex": "isFemale"})

    cleaned_data["isFemale"] = (cleaned_data["isFemale"] == "female").astype(int)

    # Cleaning age NaN

    age_mean = cleaned_data["Age"].mean().round()

    cleaned_data["Age"] = cleaned_data["Age"].fillna(age_mean)

    # Cleaning the Ticket Column

    ticket_counts = cleaned_data["Ticket"].value_counts()
    cleaned_data["Ticket Group Size"] = cleaned_data["Ticket"].map(ticket_counts)

    # Removing the Ticket Column
    cleaned_data = cleaned_data.drop("Ticket", axis="columns")

    # Cleaning Embarked

    cleaned_data["Embarked"] = cleaned_data["Embarked"].fillna("S")

    cleaned_data["Embarked"] = cleaned_data["Embarked"].astype("category").cat.codes

    # Cleaning Cabin Column

    cleaned_data["Cabin"] = cleaned_data["Cabin"].fillna("U")

    cleaned_data["Cabin"] = cleaned_data["Cabin"].astype("category").cat.codes

    # Cleaning Fare
    fare_mean = cleaned_data["Fare"].mean()
    cleaned_data["Fare"] = cleaned_data["Fare"].fillna(fare_mean)

    return cleaned_data
