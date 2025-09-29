# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("reduced_loan.csv")

# Leave clean cols
for base, alt in [
    ('verification_status', 'verification_status.1'),
    ('application_type', 'application_type.1'),
    ('initial_list_status', 'initial_list_status.1'),
    ('purpose', 'purpose.1'),
]:
      if base in df.columns and alt in df.columns:
        df[base] = df[base].fillna(df[alt])
        df = df.drop(columns=[alt])

# %%
df.head()

# %%
df.info()

# %% [markdown]
# ### 1. Exploratory Data Analysis

# %% [markdown]
# To select which variables are important to test the model and visualization

# %% [markdown]
# keep only the rows where loan_status is either "Fully Paid" or "Charged Off"

# %%
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

# %%
plt.figure(figsize=(12,6))
sns.countplot(x='loan_status', data=df)

plt.title('Distribution of Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')

plt.tight_layout() 
plt.show()


# %%
# Creating a histogram of the loan_amnt column.
plt.figure(figsize=(12,4))
sns.histplot(df['loan_amnt'], kde=False, bins=40)
plt.xlim(0, 45000)
plt.title("Distribution of Loan Amount")
plt.show()


# %% [markdown]
# Find the correlation between the feature variables and calculation

# %%
df.corr(numeric_only=True)

# %% [markdown]
# Using heatmap to visualise the correlation

# %%
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='viridis')
plt.ylim(10, 0)


# %% [markdown]
# the above heatmap shows the correlation of various variables and they are hard to define. I created the new 50 strongest correlation features

# %% [markdown]
# Create the scatter plot to visualize the relationship between monthly installment and total loan amount by the borrower

# %%
plt.figure(figsize=(12,8))
sns.scatterplot(x='installment', y='loan_amnt', data=df)


# %% [markdown]
# Create a boxplot showing the relationship between the loan_status and the Loan Amount

# %%
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
plt.xticks(rotation=30, ha='right')


# %% [markdown]
# Summarize statistics for the loan amount, grouped by the loan_status

# %%
df.groupby('loan_status')['loan_amnt'].describe()

# %% [markdown]
# Explaination:
# 
# - Charged Off:
# - Higher average loan amount.
# - These are borrowers who defaulted and failed to fully repay their loans.
# - Median also higher than Fully Paid → consistently larger loans.
# - Fully Paid:
# - Slightly lower average loan amount.
# - Borrowers successfully repaid in full.
# - Median loan amount → suggests more borrowers took smaller loans.
# - Standard Deviation (Both Groups) indicating wide variation in loan amounts for both categories. Similar spread shows diversity in loan sizes regardless of repayment status.
# - Minimum and Maximum: Range is consistent showing the loan program's limits are the same across categories.
# 
# Insights: 
# - Borrowers who defaulted (Charged Off) tend to take out bigger loans on average than those who fully repaid.
# - The distribution is right-skewed, as the mean is higher than the median in both categories — indicating some high-value outlier loans.
# - Larger loans are riskier: higher average and median in defaulted group suggests a correlation between loan size and repayment difficulty.
# - Loan repayment behavior varies with amount: borrowers with smaller loans may find it easier to repay fully.
# - Consistent variability in both groups’ loan amounts may imply other risk factors are involved beyond just loan size (e.g., credit history, income).

# %% [markdown]
# Look at Grade and SubGrade columns to attribute to the loans.

# %%
sorted(df['grade'].unique())

# %%
sorted(df['sub_grade'].unique())

# %% [markdown]
# Create a countplot per grade and set loan_status label

# %%
plt.figure(figsize=(12,7))
sns.countplot(x='grade',data=df,hue='loan_status')

# %% [markdown]
# plotting a count plot per subgrade

# %%
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )

# %% [markdown]
# loan status sub grade wise

# %%
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')

# %% [markdown]
# Create a new column called 'load_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".

# %%
df['loan_status'].unique()

# %%
df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

df[['loan_status', 'loan_repaid']].head()


# %% [markdown]
# Creating a bar plot showing the correlation of the numeric features to the new loan_repaid column

# %%
plt.figure(figsize=(12, 7))
numeric_df = df.select_dtypes(include=['number'])
corr_values = numeric_df.corr()['loan_repaid'].drop('loan_repaid')

filtered_corr = corr_values[abs(corr_values) > 0.05].sort_values()
plt.figure(figsize=(14, 6))
filtered_corr.plot(kind='bar')
plt.title("Correlation with loan_repaid", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 2. Data Pre Processing

# %% [markdown]
# Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables.

# %%
df.head()

# %% [markdown]
# ### 3.Missing Data

# %% [markdown]
# Use the factors to decide wherte the data columns would be useful in order to keep, discard or fill the missing data

# %%
len(df)

# %% [markdown]
# Create a series that displays the total count of missing values per column

# %%
df.isnull().sum()

# %% [markdown]
# Converting this Series to be in term of percentage of the total DataFrame

# %%
100*df.isnull().sum()/len(df)


# %% [markdown]
# examine emp_title and emp_length to see whether it will be okay to drop them. Print out their feature information using the feat_info()
# function from the top of this notebook

# %%
print(df.columns)


# %%
df['emp_title'].nunique()

# %%
df['emp_title'].value_counts()

# %% [markdown]
# Since there are too many job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column.

# %%
df = df.drop('emp_title',axis=1)

# %% [markdown]
# Create a count plot of the emp_length feature column.

# %%
sorted(df['emp_length'].dropna().unique())

# %%
emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']

# %%
plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=df,order=emp_length_order)

# %% [markdown]
# Plotting out the countplot with a hue separating Fully Paid vs Charged Off

# %%
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')

# %% [markdown]
# the chart does not really confirm of the strong relationship between the length and being charged off, what we need is the percentage of charge offs per category, to confirm what percentage of borrowers per employment category didn't pay back the loan.

# %%
emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']

# %%
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']

# %%
emp_len = emp_co/emp_fp

# %%
emp_len

# %%
plt.figure(figsize=(12,4))
emp_len.plot(kind='bar')

# %% [markdown]
# Thus, the charge off rates are similar across all employment lengths, the emp_length column will be dropped 

# %%
df = df.drop('emp_length',axis=1)

# %%
#Revisit the DataFrame to see what feature columns still have missing data.
df.isnull().sum()

# %%

# Review the title column vs the purpose column. Is this repeated information? 
# can be removed -- Aura
df['purpose'].head(20)

# %%
df['title'].head(20)

# %% [markdown]
# The title column is simply a string subcategory/description of the purpose column. we can go ahead and drop the title column.

# %%
df=df.drop('title',axis=1)

# %%
#Revisit the DataFrame to see what feature columns still have missing data.
df.isnull().sum()

# %%
df['mort_acc'].value_counts()

# %% [markdown]
# review the other columsn to see which most highly correlates to mort_acc

# %% [markdown]
# Convert "term" to integer

# %%
# Remove the word " months" and convert to integer
df['term'] = df['term'].astype(str).str.strip().str.replace(' months', '').astype(int)

# %%
print("Correlation with the mort_acc column")
df.corr(numeric_only=True)['mort_acc'].sort_values()

# %% [markdown]
# Looks like the total_acc feature correlates with the mort_acc , this makes sense! Let's try this fillna() approach. We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry. To get the result below:

# %%
print("Mean of mort_acc column per total_acc")
df.select_dtypes(include='number').groupby(df['total_acc']).mean()['mort_acc']

# %% [markdown]
# Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above. This involves using an .apply() method with two columns.

# %%
total_acc_avg = df.groupby('total_acc')['mort_acc'].mean()

# %%
total_acc_avg

# %%
total_acc_avg[2.0]

# %%
def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

# %%
df['mort_acc'] = df['mort_acc'].fillna(df['total_acc'].map(total_acc_avg))

# %%
df.isnull().sum()

# %% [markdown]
# revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data. so we can go ahead and remove the rows that are missing those values in those columns with dropna().

# %%
df = df.dropna()

# %%
df.isnull().sum()    

# %% [markdown]
# ### Categorical Variables and Dummy Variables

# %% [markdown]
# We're done working with the missing data! Now we just need to deal with the string values due to the categorical columns.

# %%
df.select_dtypes(['object']).columns

# %% [markdown]
# ### term feature
# Convert the term feature into either a 36 or 60 integer numeric data type using .apply() or .map().**

# %%
df['term'].value_counts()

# %%
df['term'] = df['term'].astype(str).str.replace(' months', '').astype(int)

# %% [markdown]
# ### grade feature
# We already know grade is part of sub_grade, so just drop the grade feature.

# %%
print(df.columns)

# %%
df = df.drop('grade', axis=1, errors='ignore')


# %% [markdown]
# Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe. Remember to drop the original subgrade column and to add drop_first=True to your get_dummies call.

# %%
df.columns = df.columns.str.strip()

if 'sub_grade' in df.columns:
    subgrade_dummies = pd.get_dummies(df['sub_grade'], prefix='sub_grade', drop_first=True)
    df = pd.concat([df.drop(columns=['sub_grade']), subgrade_dummies], axis=1)

# %%
df.select_dtypes(['object']).columns

# %% [markdown]
# ### verification_status, application_type,initial_list_status,purpose
# Convert these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.

# %%
print("Current columns in df:", df.columns.tolist())

# %%
# Columns we want to encode
base_cols = ['verification_status', 'application_type', 'initial_list_status', 'purpose']

# Find which of these columns actually exist in df (including possible ".1" versions)
cols_to_encode = [c for c in df.columns if c in base_cols or c.replace('.1', '') in base_cols]

# Only proceed if there are valid columns to encode
if cols_to_encode:
    # One-hot encode the selected columns
    dummies = pd.get_dummies(df[cols_to_encode], drop_first=True)

    # Drop the original categorical columns
    df = df.drop(cols_to_encode, axis=1)

    # Remove any duplicate columns from dummies (safety check)
    dummies = dummies.loc[:, ~dummies.columns.duplicated()]

    # Concatenate the encoded columns back to df
    df = pd.concat([df, dummies], axis=1)


# %%
print(df.columns.tolist())


# %% [markdown]
# ### home_ownership
# Review the value_counts for the home_ownership column.

# %%
df['home_ownership'].value_counts()

# %% [markdown]
# We need to Convert these to dummy variables, but replace NONE and ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT, OWN, OTHER. Then concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.

# %%
print(df.columns.tolist())


# %%
if 'home_ownership' in df.columns:  
    df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

    dummies = pd.get_dummies(df['home_ownership'], prefix="home_ownership", drop_first=True)

    df = pd.concat([df.drop('home_ownership', axis=1), dummies], axis=1)
else:
    print("Column 'home_ownership' not found, maybe already dummy encoded.")

# %%
df.columns

# %% [markdown]
# ### issue_d
# This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature.

# %%
df = df.drop('issue_d',axis=1)

# %% [markdown]
# ### earliest_cr_line
# This appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.

# %%
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)     

# %%
df.head(5)

# %% [markdown]
# addtionnally, funded_amnt and funded_amnt_inv columns are dropped as well.

# %%
df.head(5)

# %% [markdown]
# ### address
# Let's feature engineer a zip code column from the address in the data set. Create a column called 'zip_code' that extracts the zip code from the address column.

# %%
df['zip_code'] = df['addr_state'].apply(lambda address:address[-5:])

# %% [markdown]
# Now make this zip_code column into dummy variables using pandas. Concatenate the result and drop the original zip_code column along with dropping the address column.

# %%
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','addr_state'],axis=1)
df = pd.concat([df,dummies],axis=1)

# %%
df.columns

# %%
df.select_dtypes(['object']).columns

# %% [markdown]
# ## Train Test Split

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score

# %%
# Define KS Statistics function
def ks_statistics(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))

# %% [markdown]
# drop the load_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s.

# %%
df = df.drop('loan_status', axis =1)

# %% [markdown]
# ## Normalizing the Data (NEED To fix)
# Using a MinMaxScaler to normalize the feature data X_train and X_test. Recall we don't want data leakge from the test set so we only fit on the X_train data.

# %%
X = df.drop('loan_repaid', axis=1)
y = df['loan_repaid']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size=0.2, random_state=42, stratify=y)


# Drop or encode non-numeric features BEFORE scaling
non_numeric_cols = X_train.select_dtypes(include=['object']).columns
print(non_numeric_cols)

# Drop them if already encoded
X_train = pd.get_dummies(X_train, columns=non_numeric_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=non_numeric_cols, drop_first=True)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Scale
from sklearn.preprocessing import MinMaxScaler
scaler_mm = MinMaxScaler()
X_train_scaled_mm = scaler_mm.fit_transform(X_train)
X_test_scaled_mm = scaler_mm.transform(X_test)

# Impletement Wide & Deep

# Wide
X_train_wide = X_train.values.astype("float32")
X_test_wide = X_test.values.astype("float32")

# Deep: use the result from scaler
X_train_deep = X_train_scaled_mm.astype("float32")
X_test_deep = X_test_scaled_mm.astype("float32")


# %% [markdown]
# ### Creating the Model
# Building a sequential model to be trained on the data. We have unlimited options here, but here is what the solution uses: a model that goes 78 --> 39 --> 19--> 1 output neuron. OPTIONAL: Explore adding Dropout layers 1) 2

# %%
model = Sequential()

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw


# input layer
model.add(Dense(78, activation='relu', input_shape=(X_train_scaled_mm.shape[1],)))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[
        tf.keras.metrics.AUC(name='roc_auc', curve='ROC'),
        tf.keras.metrics.AUC(name='pr_auc',  curve='PR'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy')
    ]
)

# %% [markdown]
# ### Create Wide & Deep Model

# %%
# Use Tensorflow
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Add

# Define 2 inputs
wide_in = Input(shape=(X_train_wide.shape[1],), name="wide_input")
deep_in = Input(shape=(X_train_deep.shape[1],), name="deep_input")

# Wide branch： liner logit
wide_logit = Dense(1, activation=None, name="wide_logit")(wide_in)

# Deep Branch: MLP
x = Dense(128, activation="relu")(deep_in)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
deep_logit = Dense(1, activation=None, name="deep_logit")(x)

# Add logits, apply signmoid to get probabilities
sum_logit = Add(name="sum_logits")([wide_logit, deep_logit])
prob = Activation("sigmoid", name="prob")(sum_logit)

wd_model = Model(inputs=[wide_in, deep_in], outputs=prob, name="WideAndDeep")

# Complie model
wd_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy")
    ],
)


# get model summary
wd_model.summary()


# %% [markdown]
# Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting. Optional: add in a batch_size of 256.

# %%
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {int (c): w for c, w in zip(classes, cw)}

model.fit(
    x=X_train_scaled_mm,
    y=y_train,
    epochs=25,
    batch_size=256,
    validation_data=(X_test_scaled_mm, y_test),
    class_weight = class_weight_dict
)


# %%
# training: algin with the existing class_weight

history_wd = wd_model.fit(
    x=[X_train_wide, X_train_deep],
    y=y_train,
    epochs=25,
    batch_size=256,
    validation_data=([X_test_wide, X_test_deep], y_test),
    class_weight=class_weight_dict,
    verbose=1
)

# %% [markdown]
# ## Evaluating Model Performance.
# Plot out the validation loss versus the training loss.

# %%
feature_cols = X_train.columns.tolist()

# %%
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot(figsize=(12,7))
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.show()

# %% [markdown]
# Creating predictions from the X_test set and display a classification report and confusion matrix for the X_test set.

# %%
pred_prob = model.predict(X_test_scaled_mm).ravel()
pred_cls  = (pred_prob > 0.5).astype(int)

# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_cls))

# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred_cls)

# %%
random_index = np.random.randint(0, len(df))
new_customer = df.drop(['loan_repaid','loan_status'], axis=1, errors='ignore').iloc[[random_index]]

# %%
new_customer_encoded = pd.get_dummies(new_customer, drop_first=True)

if new_customer_encoded.columns.duplicated().any():
    new_customer_encoded = new_customer_encoded.loc[:, ~new_customer_encoded.columns.duplicated()]

train_cols = list(getattr(scaler_mm, "feature_names_in_", pd.Index(feature_cols).drop_duplicates()))

new_customer_aligned = new_customer_encoded.reindex(columns=train_cols, fill_value=0)

new_customer_scaled = scaler_mm.transform(new_customer_aligned)

pred_prob_one = model.predict(new_customer_scaled).ravel()[0]
pred_cls_one  = int(pred_prob_one > 0.5)
print("Predicted class:", pred_cls_one)


# %%
# Now check, did this person actually end up paying back their loan?
actual = df.iloc[random_index]['loan_repaid']

# %% [markdown]
# It's a false positive: the model predicted repayment (1), but actual result was default (0).

# %%
from sklearn.metrics import roc_auc_score, RocCurveDisplay
pred_prob = model.predict(X_test_scaled_mm).ravel()
ks = ks_statistics(y_test, pred_prob)
auc = roc_auc_score(y_test, pred_prob)
pr_auc = average_precision_score(y_test, pred_prob) 
RocCurveDisplay.from_predictions(y_test, pred_prob)
plt.title(f'ROC Curve (AUC={auc:.3f})')
plt.grid(True); plt.show()
print("ROC-AUC Score:", auc)

# %%
# Save Keras MLP result into results dict
results = {}

keras_probs = model.predict(X_test_scaled_mm).ravel()
keras_preds = (keras_probs > 0.5).astype(int)

results['Keras MLP (Class Weight)'] = {
    "report": classification_report(y_test, keras_preds, output_dict=True),
    "roc_auc": roc_auc_score(y_test, keras_probs),
    "pr_auc":  average_precision_score(y_test, keras_probs),
    "ks":      ks_statistics(y_test, keras_probs)
}

# %%
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], 'k--', label="Random Guessing")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from pprint import pprint

# ------------------------#
# 1. Logistic Regression Model
# ------------------------#

# Standardize the data
scaler_std = StandardScaler()
X_train_scaled_std = scaler_std.fit_transform(X_train)
X_test_scaled_std = scaler_std.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42, class_weight='balanced') # add class_weight
lr_model.fit(X_train_scaled_std, y_train)
lr_preds = lr_model.predict(X_test_scaled_std)
lr_probs = lr_model.predict_proba(X_test_scaled_std)[:, 1]
pr_auc = average_precision_score(y_test, lr_probs)

# Calculte KS
ks = ks_statistics(y_test, lr_probs)

# Store results
results['Logistic Regression (Class Weight)'] = {
    "report": classification_report(y_test, lr_preds, output_dict=True),
    "roc_auc": roc_auc_score(y_test, lr_probs),
    "pr_auc": average_precision_score(y_test, lr_probs),
    "ks": ks  #save ks
}

# Print results
print("== Logistic Regression Report ==")
pprint(results['Logistic Regression (Class Weight)']['report'])
print("ROC-AUC Score:", results['Logistic Regression (Class Weight)']['roc_auc'])

# Quick AUC print
print("Quick AUC:", roc_auc_score(y_test, lr_probs))


# %%
# 1. Select a random customer from the original DataFrame
random_index = np.random.randint(0, len(df))
new_customer = df.iloc[[random_index]].copy()

# 2. Save the actual label for comparison (if available)
actual_label = new_customer["loan_repaid"].values[0]

# 3. Drop the target column and any leakage columns not used in training
new_customer = new_customer.drop(["loan_repaid", "loan_status"], axis=1, errors="ignore")

# 4. Apply OneHotEncoding same as during training
new_customer_encoded = pd.get_dummies(new_customer, drop_first=True)

# 4.1. If there are duplicate column names after encoding, drop duplicates
if new_customer_encoded.columns.duplicated().any():
    new_customer_encoded = new_customer_encoded.loc[:, ~new_customer_encoded.columns.duplicated()]

# 5. Use the column order from the scaler (preferred) or fallback to feature_cols
train_cols = list(getattr(scaler_std, "feature_names_in_", pd.Index(feature_cols).drop_duplicates()))

# 6. Reindex to match training columns exactly: fill missing with 0, drop extras
new_customer_aligned = new_customer_encoded.reindex(columns=train_cols, fill_value=0)

# 7. Scale the data with the same scaler used on training data
#    (DataFrame with aligned column names avoids "feature names" warning)
new_customer_scaled = scaler_std.transform(new_customer_aligned)

# 8. Make prediction (example: using Logistic Regression)
predicted_class = lr_model.predict(new_customer_scaled)[0]
predicted_prob  = lr_model.predict_proba(new_customer_scaled)[0][1]

# 9. Print the results
print("== PREDICTION RESULT ==")
print(" Random Customer Index:", random_index)
print(" Prediction:", "Fully Paid (1)" if predicted_class == 1 else "Charged Off (0)")
print(" Probability of repayment:", round(predicted_prob * 100, 2), "%")
print(" Actual (if available):", "Fully Paid (1)" if actual_label == 1 else "Charged Off (0)")


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from pprint import pprint

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predict class labels and probabilities
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, rf_probs)

ks = ks_statistics(y_test, rf_probs)

# Save evaluation results into the `results` dictionary
results['Random Forest (Class Weight)'] = {
    "report": classification_report(y_test, rf_preds, output_dict=True),
    "roc_auc": roc_auc_score(y_test, rf_probs),
    "pr_auc": average_precision_score(y_test, rf_probs),
    "ks": ks
}

# Print the results
print("== Random Forest Report ==")
pprint(results['Random Forest (Class Weight)']['report'])
print("ROC-AUC Score:", results['Random Forest (Class Weight)']['roc_auc'])


# %%
# 1. Select a random customer from the original DataFrame
random_index = np.random.randint(0, len(df))
new_customer = df.iloc[[random_index]].copy()

# 2. Save the actual label for comparison (if available)
actual_label = new_customer["loan_repaid"].values[0]

# 3. Drop target and any leakage columns not used when training RF
new_customer = new_customer.drop(["loan_repaid", "loan_status"], axis=1, errors="ignore")

# 4. Apply OneHotEncoding (same as during training)
new_customer_encoded = pd.get_dummies(new_customer, drop_first=True)

# 4.1 If duplicated columns appeared after encoding, keep the first occurrence
if new_customer_encoded.columns.duplicated().any():
    new_customer_encoded = new_customer_encoded.loc[:, ~new_customer_encoded.columns.duplicated()]

# 5. Use the exact training column order that RF saw at fit time
#    (falls back to your feature_cols if the attribute is not available)
rf_train_cols = list(getattr(rf_model, "feature_names_in_", pd.Index(feature_cols).drop_duplicates()))

# 6. Reindex to match training columns exactly: fill missing with 0, drop extras
new_customer_aligned = new_customer_encoded.reindex(columns=rf_train_cols, fill_value=0)

# 7. Predict with Random Forest (no scaling for RF)
predicted_class = rf_model.predict(new_customer_aligned)[0]
predicted_prob  = rf_model.predict_proba(new_customer_aligned)[0][1]

# 8. Print the prediction results
print("== PREDICTION USING RANDOM FOREST ==")
print(" Random Customer Index:", random_index)
print(" Prediction:", "Fully Paid (1)" if predicted_class == 1 else "Charged Off (0)")
print(" Probability of Repayment:", round(predicted_prob * 100, 2), "%")
print(" Actual Outcome:", "Fully Paid (1)" if actual_label == 1 else "Charged Off (0)")

# %%
# ------------------------#
# 3. Naive Bayes
# ------------------------#
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_sample_weight

# sample weight
sw = compute_sample_weight(class_weight='balanced', y=y_train)

# Initialize and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train, sample_weight=sw)

# Make predictions
nb_preds = nb_model.predict(X_test)
nb_probs = nb_model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, nb_probs)

# Calculate ks
ks = ks_statistics(y_test, nb_probs) 

# Save evaluation results
results['Naive Bayes (Sample Weight)'] = {
    "report": classification_report(y_test, nb_preds, output_dict=True),
    "roc_auc": roc_auc_score(y_test, nb_probs),
    "pr_auc": average_precision_score(y_test, nb_probs),
    "ks": ks_statistics(y_test,nb_probs)
}

# Print the results
print("== Naive Bayes Report ==")
print(results['Naive Bayes (Sample Weight)']['report'])
print("ROC-AUC Score:", results['Naive Bayes (Sample Weight)']['roc_auc'])


# %%
# 1. Select a random customer from the original DataFrame
random_index = np.random.randint(0, len(df))
new_customer = df.iloc[[random_index]].copy()

# 2. Store the actual label for comparison
actual_label = new_customer["loan_repaid"].values[0]

# 3. Drop the target column (and leakage if present)
new_customer = new_customer.drop(["loan_repaid", "loan_status"], axis=1, errors="ignore")

# 4. Apply One-Hot Encoding as done during training
new_customer_encoded = pd.get_dummies(new_customer, drop_first=True)

# 4.1 If duplicated columns appeared, keep the first occurrence
if new_customer_encoded.columns.duplicated().any():
    new_customer_encoded = new_customer_encoded.loc[:, ~new_customer_encoded.columns.duplicated()]

# 5. Use the exact training column order seen by Naive Bayes at fit time
nb_train_cols = list(getattr(nb_model, "feature_names_in_", X_train.columns))

# 6. Align columns to match training features exactly (fill missing with 0, drop extras)
new_customer_aligned = new_customer_encoded.reindex(columns=nb_train_cols, fill_value=0)

# 7. DO NOT scale for Naive Bayes here (the model was trained on X_train, not scaled)
#    Predict using the trained Naive Bayes model
predicted_class = nb_model.predict(new_customer_aligned)[0]
predicted_prob  = nb_model.predict_proba(new_customer_aligned)[0][1]

# 8. Print the prediction result
print("==  PREDICTION USING NAÏVE BAYES ==")
print("Random Customer Index:", random_index)
print("Prediction:", "Fully Paid (1)" if predicted_class == 1 else "Charged Off (0)")
print("Repayment Probability:", round(predicted_prob * 100, 2), "%")
print("Actual Label:", "Fully Paid (1)" if actual_label == 1 else "Charged Off (0)")

# %%
from sklearn.neural_network import MLPClassifier

# Train ANN on the standardized data
ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                          early_stopping=True, random_state=42)

ann_model.fit(X_train_scaled_std, y_train)

# Predict labels and probabilities
ann_preds = ann_model.predict(X_test_scaled_std)
ann_probs = ann_model.predict_proba(X_test_scaled_std)[:, 1]
pr_auc = average_precision_score(y_test, ann_probs)
ks = ks_statistics(y_test, ann_probs)



# Save evaluation results in the `results` dictionary
results['ANN (Baseline)'] = {
    "report": classification_report(y_test, ann_preds, output_dict=True),
    "roc_auc": roc_auc_score(y_test, ann_probs),
    "pr_auc": average_precision_score(y_test, ann_probs),
    "ks": ks
}

# Display the results
print("== Artificial Neural Network Report ==")
print(results['ANN (Baseline)']['report'])
print("ROC-AUC Score:", results['ANN (Baseline)']['roc_auc'])
print("KS:", results['ANN (Baseline)']['ks'])



# %%
from imblearn.over_sampling import SMOTE

# ANN SMOTE
sm = SMOTE(random_state=42)
Xtr_sm,ytr_sm = sm.fit_resample(X_train_scaled_std, y_train)

ann_model_sm = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                          early_stopping=True, random_state=42)
ann_model_sm.fit(Xtr_sm, ytr_sm)
ann_preds_sm = ann_model_sm.predict(X_test_scaled_std)
ann_probs_sm = ann_model_sm.predict_proba(X_test_scaled_std)[:, 1]
ks_sm = ks_statistics(y_test, ann_probs_sm)

results['Artificial Neural Network (SMOTE)'] = {
    "report": classification_report(y_test, ann_preds_sm, output_dict=True),
    "roc_auc": roc_auc_score(y_test, ann_probs_sm),
    "pr_auc": average_precision_score(y_test, ann_probs_sm),
    "ks": ks_sm
}

print("== Artificial Neural Network Report(SMOTE) ==")
print(results['Artificial Neural Network (SMOTE)']['report'])
print("ROC-AUC Score:", results['Artificial Neural Network (SMOTE)']['roc_auc'])
print("KS:", results['Artificial Neural Network (SMOTE)']['ks'])

# %%
# 1. Select a random customer from the original dataframe
random_index = np.random.randint(0, len(df))
new_customer = df.iloc[[random_index]].copy()

# 2. Save the actual label for comparison
actual_label = new_customer["loan_repaid"].values[0]

# 3. Remove the target column (and drop leakage if present)
new_customer = new_customer.drop(["loan_repaid", "loan_status"], axis=1, errors="ignore")

# 4. Apply one-hot encoding as done during training
new_customer_encoded = pd.get_dummies(new_customer, drop_first=True)

# 4.1 If duplicated columns appeared after encoding, keep the first occurrence
if new_customer_encoded.columns.duplicated().any():
    new_customer_encoded = new_customer_encoded.loc[:, ~new_customer_encoded.columns.duplicated()]

# 5. Use the exact training column order the scaler saw at fit time (fallback to de-duplicated feature_cols)
train_cols = list(getattr(scaler_std, "feature_names_in_", pd.Index(feature_cols).drop_duplicates()))

# 6. Align columns to match training: fill missing with 0, drop extras
new_customer_aligned = new_customer_encoded.reindex(columns=train_cols, fill_value=0)

# 7. Standardize the data using the same scaler fitted on X_train
#    (Passing a DataFrame with aligned column names avoids the "feature names/order" error)
new_customer_scaled = scaler_std.transform(new_customer_aligned)

# 8. Make prediction using the ANN model
predicted_class = ann_model.predict(new_customer_scaled)[0]
predicted_prob  = ann_model.predict_proba(new_customer_scaled)[0][1]

# 9. Print the result
print("== PREDICTION USING ANN ==")
print("Random Customer Index:", random_index)
print("Predicted:", "Fully Paid (1)" if predicted_class == 1 else "Charged Off (0)")
print("Repayment Probability:", round(predicted_prob * 100, 2), "%")
print("Actual Label:", "Fully Paid (1)" if actual_label == 1 else "Charged Off (0)")


# %%
# Add wide and deep to result
wd_pred_prob = wd_model.predict([X_test_wide, X_test_deep], verbose=0).ravel()
wd_pred_cls = (wd_pred_prob > 0.5).astype(int)

wd_report = classification_report(y_test, wd_pred_cls, output_dict=True)
wd_roc_auc = roc_auc_score(y_test, wd_pred_prob)
wd_pr_auc = average_precision_score(y_test, wd_pred_prob)
wd_ks = ks_statistics(y_test, wd_pred_prob)

# Add to result dict
results['Wide & Deep (Class Weight)'] = {
    "report": wd_report,
    "roc_auc": wd_roc_auc,
    "pr_auc": wd_pr_auc,
    "ks": wd_ks
}

# %%

# Create a list to store evaluation results
summary_data = []

def safe_round(val, n=2):
    return round(val, n) if val is not None else None

# Iterate through all saved models in the 'results' dictionary
for model_name, metrics in results.items():
    report = metrics.get("report")
    roc_auc = metrics.get("roc_auc")
    pr_auc = metrics.get("pr_auc")
    ks_val =metrics.get("ks") 

    summary_data.append({
        "Model": model_name,
        "Accuracy": safe_round(report["accuracy"], 5),
        "Precision": safe_round(report["1"]["precision"], 5),
        "Recall": safe_round(report["1"]["recall"], 5),
        "F1-Score": safe_round(report["1"]["f1-score"], 5),
        "ROC-AUC": safe_round(roc_auc, 4),
        "PR-AUC": safe_round(pr_auc, 4),
        "KS": safe_round(ks_val, 4)
    })

# Create a DataFrame from the summary list
summary_table = pd.DataFrame(summary_data)

# Sort the table by ROC-AUC in descending order
summary_table = summary_table.sort_values(by=["PR-AUC", "ROC-AUC"], ascending=False).reset_index(drop=True)



# Display the summary table
from IPython.display import display
display(summary_table)



