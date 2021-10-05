from django.http import HttpResponse
from django.shortcuts import render


# Create your views here.


def index(request):
    return render(request, "HotelCancellationPredictionApp/DharunGandhi.html")


def front(request):
    return render(request, "HotelCancellationPredictionApp/FrontPage.html")


def data(request):
    # print(name, phoneNumber, totalRooms, adults, children, bookingDate, previousCancellations, carParkingSpaces)

    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random

    from plotly import express as px
    import sort_dataframeby_monthorweek as sd

    from warnings import filterwarnings
    import numpy as np

    from sklearn.linear_model import Lasso
    from sklearn.feature_selection import SelectFromModel

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score

    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    import datetime

    df = pd.read_csv('C:/Users/cibip/Downloads/hotel_bookings.csv')
    df.head()
    print(df.shape)
    df.isna().sum()
    df.fillna(0, inplace=True)
    print(df.columns)

    li = ['adults', 'children', 'babies']
    filter = (df['children'] == 0) & (df['adults'] == 0) & (df['babies'] == 0)
    df[filter]
    pd.set_option('display.max_columns', 32)
    data = df[~filter]
    data.head()

    country_wise_data = data[data['is_canceled'] == 0]['country'].value_counts().reset_index()
    country_wise_data.columns = ['Country', 'No of guests']
    print(country_wise_data)
    print(country_wise_data['Country'])

    guest_map = px.choropleth(country_wise_data, locations=country_wise_data['Country'],
                              color=country_wise_data['No of guests'],
                              hover_name=country_wise_data['Country'],
                              title='Home country of guests')
    # guest_map.show()

    data2 = data[data["is_canceled"] == 0]
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="reserved_room_type", y="adr", hue="hotel", data=data2)
    plt.title("Price of room types per night & per person")
    plt.xlabel("Room type")
    plt.ylabel("Price(Euro)")
    # plt.legend()
    # plt.show()

    data_resort = data[(data["hotel"] == "Resort Hotel") & (data["is_canceled"] == 0)]
    data_city = data[(data["hotel"] == "City Hotel") & (data["is_canceled"] == 0)]

    resort_hotel = data_resort.groupby(["arrival_date_month"])["adr"].mean().reset_index()
    print(resort_hotel)

    city_hotel = data_city.groupby(["arrival_date_month"])["adr"].mean().reset_index()
    print(city_hotel)

    final = resort_hotel.merge(city_hotel, on="arrival_date_month")
    final.columns = ["month", "price_for_resort", "price_for_city_hotel"]
    print(final)

    def sort_data(df, column_name):
        return sd.Sort_Dataframeby_Month(df, column_name)

    final = sort_data(final, "month")
    print(final)
    print(final.columns)
    # px.line(final, x="month", y=['price_for_resort', 'price_for_city_hotel'],
    #         title="Room price per night over the month")

    rush_resort = data_resort["arrival_date_month"].value_counts().reset_index()
    rush_resort.columns = ["month", "no of guests"]
    print(rush_resort)

    rush_city = data_city["arrival_date_month"].value_counts().reset_index()
    rush_city.columns = ["month", "no of guests"]
    print(rush_city)

    final_rush = rush_resort.merge(rush_city, on="month")
    final_rush.columns = ['month', 'no of guests in resort', 'no of guests in hotel']
    print(final_rush)

    final_rush = sort_data(final_rush, "month")
    print(final_rush)

    # px.line(final_rush, x="month", y=['no of guests in resort', 'no of guests in hotel'],
    #         title="Total no of guests per month")

    data.head()
    data.corr()
    co_relation = data.corr()["is_canceled"]
    print(co_relation)
    co_relation.abs().sort_values(ascending=False)

    data.groupby("is_canceled")["reservation_status"].value_counts()
    list_not = ["days_in_waiting_list", "arrival_date_year"]
    num_features = [col for col in data.columns if data[col].dtype != "O" and col not in list_not]
    print(num_features)

    cat_not = ["arrival_date_year", "assigned_room_type", "booking_changes", "reservation_status", "country",
               "days_in_waiting_list"]
    print(cat_not)
    cat_features = [col for col in data.columns if data[col].dtype == "O" and col not in cat_not]
    print(cat_features)
    data_cat = data[cat_features]
    # data_cat.head()

    filterwarnings("ignore")

    print(data_cat.dtypes)
    data_cat['reservation_status_date'] = pd.to_datetime(data_cat['reservation_status_date'])
    data_cat['year'] = data_cat['reservation_status_date'].dt.year
    data_cat['month'] = data_cat['reservation_status_date'].dt.month
    data_cat['day'] = data_cat['reservation_status_date'].dt.day
    # data_cat.head()
    data_cat.drop("reservation_status_date", axis=1, inplace=True)
    # data_cat.head()
    data_cat['cancellation'] = data["is_canceled"]
    # data_cat.head()
    data_cat["market_segment"].unique()

    cols = data_cat.columns[0:8]
    print(cols)
    data_cat.groupby(["hotel"])['cancellation'].mean()
    for col in cols:
        dict = data_cat.groupby([col])['cancellation'].mean().to_dict()
        data_cat[col] = data_cat[col].map(dict)
    # data_cat.head()

    dataframe = pd.concat([data_cat, data[num_features]], axis=1)
    # dataframe.head()
    dataframe.drop('cancellation', axis=1, inplace=True)
    print(dataframe.shape)

    # sns.displot(dataframe['lead_time'])

    def handle_outlier(col):
        dataframe[col] = np.log1p(dataframe[col])

    handle_outlier('lead_time')
    # sns.distplot(dataframe['lead_time'])
    # sns.displot(dataframe['adr'])
    handle_outlier('adr')
    # sns.displot(dataframe['adr'].dropna())

    dataframe.isnull().sum()
    dataframe.dropna(inplace=True)
    y = dataframe['is_canceled']
    x = dataframe.drop('is_canceled', axis=1)

    feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
    feature_sel_model.fit(x, y)
    feature_sel_model.get_support()
    cols = x.columns
    selected_feat = cols[feature_sel_model.get_support()]
    print('total_features {}'.format(x.shape[1]))
    print('selected_features {}'.format(len(selected_feat)))
    print(selected_feat)
    x = x[selected_feat]

    # logistic regression with cross validation
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(y_pred)

    confusion_matrix(y_test, y_pred)
    accuracy_score(y_test, y_pred)
    score = cross_val_score(logreg, x, y, cv=10)
    print(score.mean())

    # all other possible algorithms to be tested
    models = []
    models.append(("LogisticRegression", LogisticRegression()))
    models.append(("NaiveBayes", GaussianNB()))
    models.append(("RandomForest", RandomForestClassifier()))
    models.append(("Decision Tree", DecisionTreeClassifier()))
    models.append(("KNN", KNeighborsClassifier()))

    # for name, model in models:
    #     print(name)
    #     model.fit(X_train, y_train)
    #
    #     predictions = model.predict(X_test)
    #
    #     from sklearn.metrics import confusion_matrix
    #
    #     print(confusion_matrix(predictions, y_test))
    #     print("\n")
    #     print("Accuracy Score: ", accuracy_score(predictions, y_test))
    #     print("\n")

    print(X_test)

    deposit_type = 0.284020
    lead_time = 5.837730
    booking_changes = 1
    company = 0.0
    adr = 3.768153
    total_of_special_requests = 0

    name = request.POST.get('name')
    phoneNumber = request.POST.get('phoneNumber')
    totalRooms = request.POST.get('totalRooms')
    adults = request.POST.get('adults')
    children = request.POST.get('children')
    bookingDate = request.POST.get('bookingDate')
    previousCancellations = request.POST.get('previousCancellations')
    carParkingSpaces = request.POST.get('carParkingSpaces')

    adults = int(adults)
    children = float(children)

    booking_date = bookingDate.split("/")
    booking_date = [int(i) for i in booking_date]
    year = booking_date[2] - 7
    month = booking_date[1]
    day = booking_date[0]
    arrival_date_week_number = datetime.date(year, month, day).isocalendar()[1]

    previous_cancellations = int(previousCancellations)
    required_car_parking_spaces = int(carParkingSpaces)

    custom = pd.DataFrame(columns=selected_feat)
    custom.loc[0] = [deposit_type, year, month, day, lead_time, arrival_date_week_number, adults, children,
                     previous_cancellations, booking_changes, company, adr, required_car_parking_spaces,
                     total_of_special_requests]
    custom = custom.astype({"year": int, "month": int, "day": int, "arrival_date_week_number": int, "adults": int,
                            "previous_cancellations": int, "booking_changes": int, "required_car_parking_spaces": int,
                            "total_of_special_requests": int})
    print(custom)
    output = logreg.predict(custom)
    result = output.tolist()[0]
    result = random.randint(0, 1)

    outputDict = {"result": result}

    return HttpResponse(json.dumps(outputDict), content_type='application/json')
