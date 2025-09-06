import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


data = pd.read_csv(r"C:\\Users\\Dosya\\Desktop\\univer\\EPLanalysis.csv")
pd.set_option('display.max_columns', None)


data["attendance"] = pd.to_numeric(data["attendance"].astype(str).str.replace(",", ""), errors="coerce")
data.dropna(subset=["attendance"], inplace=True)
data["date"] = pd.to_datetime(data["date"], errors="coerce", dayfirst=True) 
data["date"] = data["date"].dt.strftime("%d-%m-%Y")

numerical_columns = [
    "Goals Home", "Away Goals", "home_possessions", "away_possessions",
    "home_shots", "away_shots", "home_on", "away_on", "home_off", "away_off",
    "home_blocked", "away_blocked", "home_pass", "away_pass", "home_chances", 
    "away_chances", "home_corners", "away_corners", "home_offside", "away_offside",
    "home_tackles", "away_tackles", "home_duels", "away_duels", "home_saves", 
    "away_saves", "home_fouls", "away_fouls", "home_yellow", "away_yellow",
    "home_red", "away_red"
]

for column in numerical_columns:
    if column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].str.replace(",", ".", regex=False)
        data[column] = pd.to_numeric(data[column], errors="coerce")

data["result"] = np.where(
    data["Goals Home"] > data["Away Goals"], "Home Win",
    np.where(data["Goals Home"] < data["Away Goals"], "Away Win", "Draw")
)

def calculate_points(row):
    if row["Goals Home"] > row["Away Goals"]:
        return pd.Series([3, 0])  
    elif row["Goals Home"] < row["Away Goals"]:
        return pd.Series([0, 3])
    else:
        return pd.Series([1, 1])

data[["Home points", "Away points"]] = data.apply(calculate_points, axis=1)

stadiumToTeam = {
    "Emirates Stadium": "Arsenal"
    , "Villa Park": "Aston Villa",
    "Gtech Community Stadium": "Brentford", 
    "Stamford Bridge": "Chelsea",
    "Selhurst Park": "Crystal Palace", 
    "Goodison Park": "Everton",
    "Elland Road": "Leeds United", 
    "The King Power Stadium": "Leicester City",
    "Old Trafford": "Manchester United", 
    "St. Mary's Stadium": "Southampton",
    "Amex Stadium": "Brighton & Hove Albion", 
    "St James' Park": "Newcastle United",
    "London Stadium": "West Ham United", 
    "Etihad Stadium": "Manchester City",
    "Tottenham Hotspur Stadium": "Tottenham Hotspur", 
    "Vitality Stadium": "Bournemouth",
    "Craven Cottage": "Fulham", 
    "Anfield": "Liverpool",
    "Molineux": "Wolverhampton Wanderers", 
    "The City Ground": "Nottingham Forest",
    "Carrow Road": "Norwich City", 
    "Turf Moor": "Burnley",
    "Vicarage Road": "Watford", 
    "Bramall Lane": "Sheffield United",
    "The Hawthorns": "West Bromwich Albion", 
    "8 Stadium": "Richmond"
}

team_logos = {
    "Arsenal": "https://resources.premierleague.com/premierleague/badges/50/t3.png",
    "Aston Villa": "https://resources.premierleague.com/premierleague/badges/50/t7.png",
    "Brentford": "https://resources.premierleague.com/premierleague/badges/50/t94.png",
    "Chelsea": "https://resources.premierleague.com/premierleague/badges/50/t8.png",
    "Crystal Palace": "https://resources.premierleague.com/premierleague/badges/50/t31.png",
    "Everton": "https://resources.premierleague.com/premierleague/badges/50/t11.png",
    "Leeds United": "https://resources.premierleague.com/premierleague/badges/50/t2.png",
    "Leicester City": "https://resources.premierleague.com/premierleague/badges/50/t13.png",
    "Manchester United": "https://resources.premierleague.com/premierleague/badges/50/t1.png",
    "Southampton": "https://resources.premierleague.com/premierleague/badges/50/t20.png",
    "Brighton & Hove Albion": "https://resources.premierleague.com/premierleague/badges/50/t36.png",
    "Newcastle United": "https://resources.premierleague.com/premierleague/badges/50/t4.png",
    "West Ham United": "https://resources.premierleague.com/premierleague/badges/50/t21.png",
    "Manchester City": "https://resources.premierleague.com/premierleague/badges/50/t43.png",
    "Tottenham Hotspur": "https://resources.premierleague.com/premierleague/badges/50/t6.png",
    "Bournemouth": "https://resources.premierleague.com/premierleague/badges/50/t91.png",
    "Fulham": "https://resources.premierleague.com/premierleague/badges/50/t54.png",
    "Liverpool": "https://resources.premierleague.com/premierleague/badges/50/t14.png",
    "Wolverhampton Wanderers": "https://resources.premierleague.com/premierleague/badges/50/t39.png",
    "Nottingham Forest": "https://resources.premierleague.com/premierleague/badges/50/t17.png",
    "Norwich City": "https://resources.premierleague.com/premierleague/badges/50/t45.png",
    "Burnley": "https://resources.premierleague.com/premierleague/badges/50/t90.png",
    "Watford": "https://resources.premierleague.com/premierleague/badges/50/t57.png",
    "Sheffield United": "https://resources.premierleague.com/premierleague/badges/50/t49.png",
    "West Bromwich Albion": "https://resources.premierleague.com/premierleague/badges/50/t35.png",
    "Richmond": "https://resources.premierleague.com/premierleague/badges/50/t35.png"
}

stadium_capacities = {
    "Emirates Stadium": 60704, 
    "Villa Park": 42682, 
    "Gtech Community Stadium": 17250,
    "Stamford Bridge": 40341, 
    "Selhurst Park": 25486, 
    "Goodison Park": 39572,
    "Elland Road": 37890, 
    "The King Power Stadium": 32261, 
    "Old Trafford": 74310,
    "St. Mary's Stadium": 32384, 
    "Amex Stadium": 31800, 
    "St James' Park": 52305,
    "London Stadium": 60000, 
    "Etihad Stadium": 53400, 
    "Tottenham Hotspur Stadium": 62850,
    "Vitality Stadium": 11307, 
    "Craven Cottage": 25700, 
    "Anfield": 54074,
    "Molineux": 32050, 
    "The City Ground": 30445, 
    "Carrow Road": 27244,
    "Turf Moor": 21944, 
    "Vicarage Road": 22200, 
    "Bramall Lane": 32050,
    "The Hawthorns": 26850, 
    "8 Stadium": 24800
}

data["Home_Team_Name"] = data["stadium"].map(stadiumToTeam)
data["Stadium_Capacity"] = data["stadium"].map(stadium_capacities)

stadium_avg_attendance = data[data["attendance"] > 0].groupby("stadium")["attendance"].mean()

def fill_attendance(row):
    if row["attendance"] > 0:
        return row["attendance"]
    if row["stadium"] in stadium_avg_attendance and not np.isnan(stadium_avg_attendance[row["stadium"]]):
        return stadium_avg_attendance[row["stadium"]]
    elif row["Stadium_Capacity"] > 0:
        return row["Stadium_Capacity"] * 0.89
    else:
        return 0

data["attendance"] = data.apply(fill_attendance, axis=1)
data["Percent_Filled"] = (data["attendance"] / data["Stadium_Capacity"]) * 100
data["Percent_Filled"] = data["Percent_Filled"].clip(0, 110)

df = data.copy()


print("Getting the data prepared for the Machine Learning")

y = df['result']
X = df.drop(['result', 'Goals Home', 'Away Goals', 'Home points', 'Away points', 
             'date', 'clock', 'calss', 'links'], axis=1, errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"The size of the training shape: {X_train.shape}")
print(f"The size of the test shape: {X_test.shape}")

categorical_features = ['Home_Team_Name', 'stadium']
numerical_features = numerical_columns + ['attendance', 'Percent_Filled', 'Stadium_Capacity']

categorical_features = [col for col in categorical_features if col in X_train.columns]
numerical_features = [col for col in numerical_features if col in X_train.columns]

print(f"Categorical Features: {categorical_features}")
print(f"Numerical Features: {len(numerical_features)}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print("Just picked the parameters")
print(f"\nThe best parameters: {search.best_params_}")
print(f"The best precision of cv: {search.best_score_:.3f}")

best_predictions = search.best_estimator_.predict(X_test)
test_accuracy = accuracy_score(y_test, best_predictions)
print(classification_report(y_test, best_predictions))

print("Confusion_matrix:")
cm = confusion_matrix(y_test, best_predictions)
print(cm)


fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Away Win', 'Draw', 'Home Win'],
            yticklabels=['Away Win', 'Draw', 'Home Win'], ax=ax1)
ax1.set_title('Confusion Matrix')
ax1.set_ylabel('Real values')
ax1.set_xlabel('Predicted Values')


if hasattr(search.best_estimator_.named_steps['classifier'], 'feature_importances_'):
    feature_names = []
    feature_names.extend(numerical_features)
    
    cat_processor = search.best_estimator_.named_steps['preprocessor'].named_transformers_['cat']
    if hasattr(cat_processor, 'get_feature_names_out'):
        cat_features = cat_processor.get_feature_names_out(categorical_features)
        feature_names.extend(cat_features)
    
    importances = search.best_estimator_.named_steps['classifier'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax2)
    ax2.set_title('Top-15 Most Important Features')
    ax2.set_xlabel('Importance')

plt.tight_layout()
plt.show()


fig2, ax3 = plt.subplots(figsize=(16, 10))

all_teams_points = {}
for idx, row in df.iterrows():
    home_team = row['Home_Team_Name']
    home_pts = row['Home points']
    
    if home_team not in all_teams_points:
        all_teams_points[home_team] = 0
    all_teams_points[home_team] += home_pts

total_points = pd.Series(all_teams_points).sort_values(ascending=False)
average_points = total_points.mean()

for i, (team, points) in enumerate(total_points.items()):
    if team in team_logos:
        try:
            response = requests.get(team_logos[team], timeout=5)
            img = plt.imread(BytesIO(response.content))
            imagebox = OffsetImage(img, zoom=0.06)
            ab = AnnotationBbox(imagebox, (i, points), frameon=False, pad=0)
            ax3.add_artist(ab)
        except Exception as e:
            ax3.text(i, points, team[:3], ha='center', va='center', 
                    fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
    else:
        ax3.text(i, points, team[:3], ha='center', va='center', 
                fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

ax3.axhline(y=average_points, color='red', linestyle='--', alpha=0.8, 
           linewidth=2, label=f'League Average: {average_points:.1f} points')
ax3.set_title('Premier League: Total Points Earned by Each Team', fontsize=16, pad=20)
ax3.set_xlabel('Teams', fontsize=12)
ax3.set_ylabel('Total Points', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(alpha=0.2, linestyle='--')
ax3.set_xticks(range(len(total_points)))
ax3.set_xticklabels(total_points.index, rotation=45, ha='right')

for i, (team, points) in enumerate(total_points.items()):
    ax3.annotate(f'{points}', (i, points), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()


print(f"League Average Points: {average_points:.1f}")
print(f"Top Team: {total_points.index[0]} - {total_points.iloc[0]} points")
print(f"Bottom Team: {total_points.index[-1]} - {total_points.iloc[-1]} points")
