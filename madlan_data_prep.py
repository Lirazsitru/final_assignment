import pandas as pd
from datetime import datetime
import numpy as np

def prepare_data(data):
    data=data.copy()

    def start():
        nonlocal data
        
      
    
        # Drop NaN values and remove "בנה ביתך"
        data = data.dropna(subset="price")
        data = data[data["price"] != "בנה ביתך"]
        data['price'] = data['price'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(float)

        ## טיפול בעמודת השטח

        # Replace specific string with NaN
        data.loc[:, 'Area'].replace("עסקאות באיזור (1000)", np.nan, inplace=True)

        ##########################################################################################################################################3

        #המרה של ערך השטח לערכים מספריים
        #הפקודה משאירה רק את הערך המספרי, ובמקרה ויש ערך חסר היא מכניסה את הערך 0
        #בנוסף אם יש ערך מסוג nan היא מכניסה לו ערך 0
        data['Area'] = data['Area'].astype('str').str.extractall('(\d+)').unstack().fillna(0).sum(axis=1).astype(float)
        data['Area'].replace(np.nan, 0, inplace=True)

        data['City'] = data['City'].replace('נהרייה', 'נהריה')

        # הגדרת תבנית להסרת סימנים מיוחדים ושליפת שם הרחוב בלבד
        pattern = r'[^\w\s]+'
        col_list =['City', 'Street', 'city_area', 'description ']

        for col in col_list:
            data[col] = data[col].str.replace(pattern, '', regex=True)
            data[col] = data[col].str.replace('/n', '')
            data[col] = data[col].str.replace('\n', '')
            data[col] = data[col].str.strip()

        pattern = r'(\d+(\.\d+)?)'
        data['room_number'] = data['room_number'].astype(str).str.extract(pattern)[0].astype(float)
        data['room_number'].replace(np.nan, 0, inplace=True)

        pattern = r'[\[\]\'\\0-9]|^\s+|\s+$'
        data['Street'] = data['Street'].astype(str).str.replace(pattern, '', regex=True).str.strip()

        data.reset_index(drop=True, inplace=True)

        data = data.copy()

        data['floor'] = np.nan
        data['total_floors'] = np.nan

        for i in range(len(data)):
            if pd.notnull(data['floor_out_of'][i]):
                split = data['floor_out_of'][i].split()
                if len(split) > 1:
                    if len(split) == 4:
                        if split[1] =='קרקע':
                            data.loc[i, 'floor'] = 0
                            data.loc[i, 'total_floors'] = split[3]
                        elif split[1] =='מרתף':
                            data.loc[i, 'floor'] = -1
                            data.loc[i, 'total_floors'] = split[3]
                        else:
                            data.loc[i, 'floor'] = split[1]
                            data.loc[i, 'total_floors'] = split[3]
                    elif len(split) == 2:
                        if split[1] == 'קרקע':
                            data.loc[i, 'floor'] = 0
                            data.loc[i, 'total_floors'] = 0
                        elif split[1] == 'מרתף':
                            data.loc[i, 'floor'] = -1
                            data.loc[i, 'total_floors'] = 0
                        else:
                            data.loc[i, 'floor'] = split[1]
                            data.loc[i, 'total_floors'] = np.nan
                    else:
                        data.loc[i, 'floor'] = np.nan
                        data.loc[i, 'total_floors'] = np.nan
                
        return data
    data = start()

    def categorize_dates(date):
        if isinstance(date, str):
            date = date.strip()

            if date == "גמיש":
                return 'flexible'
            elif date == 'מיידי':
                return 'less_than_6_months'
            elif date == "לא צויין":
                return 'not_defined'
            else:
                try:
                    date_time_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    return 'invalid_date'
        elif isinstance(date, datetime):
            date_time_obj = date
        else:
            return 'invalid_date'

        current_date = datetime.now()

        if (date_time_obj.day - current_date.day) > 365:
            return "above_year"
        else:
            if (date_time_obj.day - current_date.day) < (30 * 6):
                return 'less_than_6_months'
            elif (6 * 30) <= (date_time_obj.day - current_date.day) <= (365):
                return 'months_6_12'

    data['entranceDate '] = data['entranceDate '].apply(categorize_dates)

    def convert():
        nonlocal data
        data = data.copy()
        columns_to_convert = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ',
                              'hasBalcony ', 'hasMamad ', 'handicapFriendly ']

        mapping = {"כן": "TRUE", "לא": "FALSE", "yes": "TRUE", "no": "FALSE", "nan": "FALSE", "False": "FALSE",
                   "True": "TRUE", "יש": "TRUE", "אין": "FALSE", "יש מעלית": "TRUE", "אין מעלית": "FALSE",
                   "אין סורגים": "FALSE", "יש סורגים": "TRUE", "אין מחסן": "FALSE", "יש מחסן": "TRUE",
                   "יש מרפסת": "TRUE", "אין מרפסת": "FALSE", "יש ממ״ד": "TRUE", "אין ממ״ד": "FALSE",
                   "יש ממ\"ד": "TRUE", "אין ממ\"ד": "FALSE", "יש חנייה": "TRUE", "יש חניה": "TRUE",
                   "אין חניה": "FALSE", "יש מיזוג אוויר": "TRUE", "אין מיזוג אוויר": "FALSE",
                   "יש מיזוג אויר": "TRUE", "אין מיזוג אויר": "FALSE", "נגיש לנכים": "TRUE",
                   "לא נגיש לנכים": "FALSE", "נגיש": "TRUE", "לא נגיש": "FALSE"}

        data[columns_to_convert] = data[columns_to_convert].applymap(lambda x: mapping.get(x, x))

        columns_to_convert = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ',
                              'hasBalcony ', 'hasMamad ', 'handicapFriendly ']

        mapping = {"TRUE": 1, True: 1, "FALSE": 0, False: 0}

        data[columns_to_convert] = data[columns_to_convert].applymap(lambda x: mapping.get(x, x))

        return data

    data = convert()

    def remove_city_outliers(df, multiplier):
        city_stats = df.groupby('City')['price'].describe()

        df_no_outliers = pd.DataFrame(columns=df.columns)

        for city in city_stats.index:
            Q1 = city_stats.loc[city, '25%']
            Q3 = city_stats.loc[city, '75%']
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            df_city = df[(df['City'] == city) & (df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

            df_no_outliers = pd.concat([df_no_outliers, df_city])

        return df_no_outliers

    df_no_outliers = remove_city_outliers(data, 2)

    data = df_no_outliers.copy()

    def remove_area_outliers(df, multiplier):
        city_stats = df.groupby('City')['Area'].describe()

        df_no_outliers = pd.DataFrame(columns=df.columns)

        for city in city_stats.index:
            Q1 = city_stats.loc[city, '25%']
            Q3 = city_stats.loc[city, '75%']
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            df_city = df[(df['City'] == city) & (df['Area'] >= lower_bound) & (df['Area'] <= upper_bound)]

            df_no_outliers = pd.concat([df_no_outliers, df_city])

        return df_no_outliers

    df_no_outliers = remove_area_outliers(data, 2)

    data = df_no_outliers.copy()

    def ml(data):
        data.drop_duplicates(subset=['City', 'type', 'room_number', 'Area', 'Street', 'number_in_street', 'city_area'],
                             inplace=True)

        columns_to_replace = ['num_of_images', 'hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ',
                               'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ', 'floor',
                               'total_floors']
        data[columns_to_replace] = data[columns_to_replace].fillna(0)

        columns = ['number_in_street', 'floor', 'total_floors']
        data[columns] = data[columns].apply(pd.to_numeric, errors='coerce')

        data['condition '] = data['condition '].replace({'None': 'לא צויין', False: 'לא צויין', 'Nan': 'לא צויין'})
        data['condition '] = data['condition '].fillna('לא צויין')

        data['publishedDays '] = data['publishedDays '].replace({'60+': 60, 'Nan': -1, 'None': -1, '-': -1,
                                                                 'חדש': 1, 'חדש!': 1})
        data['publishedDays '] = data['publishedDays '].fillna(0)

        data = data[~data['type'].isin(['מגרש', 'נחלה'])]

        average_area_by_room_type = data.groupby(['room_number', 'type'])['Area'].mean()

        mask = (data['Area'].isnull()) | (data['Area'] == 0)

        room_number_type = data.loc[mask, ['room_number', 'type']]

        updated_area = room_number_type.join(average_area_by_room_type, on=['room_number', 'type'], how='left')

        data.loc[mask, 'Area'] = updated_area['Area'].round(0)

        data['Area'].isna().sum()

        average_area_by_roomnumber = data.groupby('room_number')['Area'].mean()

        mask = (data['Area'].isnull()) | (data['Area'] == 0)

        roomnumber_values = data.loc[mask, 'room_number']

        updated_area = roomnumber_values.map(average_area_by_roomnumber)

        data.loc[mask, 'room_number'] = updated_area.round(0)

        average_area_by_room_type = data.groupby(['Area', 'type'])['room_number'].mean()

        mask = (data['room_number'].isnull()) | (data['room_number'] == 0)

        area_type = data.loc[mask, ['Area', 'type']]

        updated_roomnum = area_type.join(average_area_by_room_type, on=['Area', 'type'], how='left')

        data.loc[mask, 'room_number'] = updated_roomnum['room_number'].round(0)

        average_room_type_by_area = data.groupby('Area')['room_number'].mean()

        mask = (data['room_number'].isnull()) | (data['room_number'] == 0)

        area_values = data.loc[mask, 'Area']

        updated_roomnum1 = area_values.map(average_room_type_by_area)

        data.loc[mask, 'room_number'] = updated_roomnum1.round(0)

        data = data[data['room_number'] <= 10]
        data = data[data['room_number'] != 0]

        return data

    data = ml(data)
    
    data = data[['City', 'type', 'room_number', 'Area', 'hasElevator ', 'price',
           'hasParking ', 'hasStorage ', 'condition ','hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']]

    return data

