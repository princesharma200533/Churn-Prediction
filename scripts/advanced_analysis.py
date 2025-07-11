# Advanced Analysis and Visualizations for Churn Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

print("🔬 Advanced Analysis for Churn Prediction")
print("=" * 50)

# This script assumes the main analysis has been run and variables are available
# In a real Jupyter notebook, this would be in the same notebook

def create_advanced_visualizations(df, churn_probabilities, y_test, y_pred_proba):
    """Create advanced visualizations for churn analysis"""
    
    plt.figure(figsize=(20, 15))
    
    # 1. Churn Distribution by Features
    plt.subplot(3, 4, 1)
    churn_by_contract = df.groupby(['ContractLength', 'Churn']).size().unstack()
    churn_by_contract.plot(kind='bar', ax=plt.gca())
    plt.title('Churn by Contract Length')
    plt.xlabel('Contract Length (months)')
    plt.ylabel('Number of Customers')
    plt.legend(title='Churn')
    plt.xticks(rotation=0)
    
    # 2. Monthly Charges Distribution
    plt.subplot(3, 4, 2)
    df[df['Churn'] == 'No']['MonthlyCharges'].hist(alpha=0.7, label='No Churn', bins=30)
    df[df['Churn'] == 'Yes']['MonthlyCharges'].hist(alpha=0.7, label='Churn', bins=30)
    plt.title('Monthly Charges Distribution')
    plt.xlabel('Monthly Charges ($)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 3. Data Usage vs Churn
    plt.subplot(3, 4, 3)
    df[df['Churn'] == 'No']['DataUsage'].hist(alpha=0.7, label='No Churn', bins=30)
    df[df['Churn'] == 'Yes']['DataUsage'].hist(alpha=0.7, label='Churn', bins=30)
    plt.title('Data Usage Distribution')
    plt.xlabel('Data Usage (GB)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 4. Payment Method vs Churn
    plt.subplot(3, 4, 4)
    payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index')
    payment_churn.plot(kind='bar', ax=plt.gca())
    plt.title('Churn Rate by Payment Method')
    plt.xlabel('Payment Method')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.legend(title='Churn')
    
    # 5. ROC Curve
    plt.subplot(3, 4, 5)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # 6. Precision-Recall Curve
    plt.subplot(3, 4, 6)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    # 7. Churn Probability Distribution
    plt.subplot(3, 4, 7)
    plt.hist(churn_probabilities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.title('Churn Probability Distribution')
    plt.xlabel('Churn Probability')
    plt.ylabel('Number of Customers')
    plt.legend()
    
    # 8. Feature Correlation Heatmap
    plt.subplot(3, 4, 8)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=plt.gca())
    plt.title('Feature Correlation Matrix')
    
    # 9. Call Duration vs Monthly Charges (Scatter)
    plt.subplot(3, 4, 9)
    colors = ['blue' if x == 'No' else 'red' for x in df['Churn']]
    plt.scatter(df['CallDuration'], df['MonthlyCharges'], c=colors, alpha=0.6)
    plt.xlabel('Call Duration (minutes)')
    plt.ylabel('Monthly Charges ($)')
    plt.title('Call Duration vs Monthly Charges')
    
    # 10. Data Usage vs Total Charges (Scatter)
    plt.subplot(3, 4, 10)
    plt.scatter(df['DataUsage'], df['TotalCharges'], c=colors, alpha=0.6)
    plt.xlabel('Data Usage (GB)')
    plt.ylabel('Total Charges ($)')
    plt.title('Data Usage vs Total Charges')
    
    # 11. Churn Rate by Quartiles
    plt.subplot(3, 4, 11)
    df['MonthlyCharges_Quartile'] = pd.qcut(df['MonthlyCharges'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    quartile_churn = df.groupby('MonthlyCharges_Quartile')['Churn'].apply(lambda x: (x == 'Yes').mean())
    quartile_churn.plot(kind='bar', ax=plt.gca(), color='coral')
    plt.title('Churn Rate by Monthly Charges Quartile')
    plt.xlabel('Monthly Charges Quartile')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=0)
    
    # 12. Model Confidence Distribution
    plt.subplot(3, 4, 12)
    high_confidence = churn_probabilities[(churn_probabilities < 0.2) | (churn_probabilities > 0.8)]
    medium_confidence = churn_probabilities[(churn_probabilities >= 0.2) & (churn_probabilities <= 0.8)]
    
    plt.bar(['High Confidence', 'Medium Confidence'], 
            [len(high_confidence), len(medium_confidence)], 
            color=['green', 'orange'])
    plt.title('Model Confidence Distribution')
    plt.ylabel('Number of Predictions')
    
    plt.tight_layout()
    plt.show()

def generate_business_recommendations(at_risk_customers):
    """Generate actionable business recommendations"""
    
    print("\n💼 BUSINESS RECOMMENDATIONS")
    print("=" * 50)
    
    # Analyze at-risk customer characteristics
    avg_monthly_charges = at_risk_customers['MonthlyCharges'].mean()
    avg_data_usage = at_risk_customers['DataUsage'].mean()
    common_contract = at_risk_customers['ContractLength'].mode().iloc[0]
    common_payment = at_risk_customers['PaymentMethod'].mode().iloc[0]
    
    print(f"📊 At-Risk Customer Profile:")
    print(f"   • Average Monthly Charges: ${avg_monthly_charges:.2f}")
    print(f"   • Average Data Usage: {avg_data_usage:.2f} GB")
    print(f"   • Most Common Contract Length: {common_contract} months")
    print(f"   • Most Common Payment Method: {common_payment}")
    
    print(f"\n🎯 Recommended Actions:")
    print(f"   1. RETENTION CAMPAIGNS:")
    print(f"      • Target {len(at_risk_customers)} high-risk customers immediately")
    print(f"      • Focus on customers with probability > 0.7 ({len(at_risk_customers[at_risk_customers['Churn_Probability'] > 0.7])} customers)")
    
    print(f"\n   2. PRICING STRATEGY:")
    if avg_monthly_charges > 70:
        print(f"      • Consider offering discounts to high-paying at-risk customers")
        print(f"      • Create loyalty programs for customers paying > $70/month")
    
    print(f"\n   3. CONTRACT OPTIMIZATION:")
    if common_contract == 12:
        print(f"      • Incentivize longer contract terms (24-36 months)")
        print(f"      • Offer contract upgrade bonuses")
    
    print(f"\n   4. PAYMENT METHOD IMPROVEMENTS:")
    if common_payment == 'Electronic Check':
        print(f"      • Encourage migration to more stable payment methods")
        print(f"      • Offer incentives for credit card or bank transfer setup")
    
    print(f"\n   5. USAGE-BASED INTERVENTIONS:")
    if avg_data_usage < 3:
        print(f"      • Engage low-usage customers with usage tutorials")
        print(f"      • Offer data usage incentives or free trials")

# Example usage (this would be integrated into the main notebook)
print("📈 Advanced analysis functions created successfully!")
print("💡 Use create_advanced_visualizations() and generate_business_recommendations() functions")
print("   in your main analysis after running the churn prediction model.")
