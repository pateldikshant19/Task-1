import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_wine_quality_dashboard():
    try:
        # Load data
        df = pd.read_csv('archive/WineQT.csv')
        print("Dataset loaded successfully!")
        print(f"Dataset Shape: {df.shape}")
        
        # Remove Id column if exists
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
            
        # Features and target
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Create essential prediction dashboard
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle('Wine Quality Prediction Dashboard', fontsize=18, fontweight='bold')
        
        # 1. Quality Distribution
        plt.subplot(2, 3, 1)
        df['quality'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title('Quality Distribution')
        plt.xlabel('Quality Score')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # 2. Confusion Matrix
        plt.subplot(2, 3, 2)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': 0.8})
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Quality')
        plt.ylabel('Actual Quality')
        
        # 3. Feature Importance
        plt.subplot(2, 3, 3)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(8)
        
        plt.barh(range(len(feature_importance)), feature_importance['importance'], color='green')
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Importance')
        plt.title('Top Features')
        plt.gca().invert_yaxis()
        
        # 4. Alcohol vs Quality
        plt.subplot(2, 3, 4)
        sns.boxplot(data=df, x='quality', y='alcohol', palette='viridis')
        plt.title('Alcohol Content by Quality')
        plt.xlabel('Quality Score')
        plt.ylabel('Alcohol Content')
        
        # 5. Model Performance
        plt.subplot(2, 3, 5)
        plt.axis('off')
        performance_text = f"""MODEL PERFORMANCE

Accuracy: {accuracy:.3f}
Samples: {len(X_test)}
Features: {len(X.columns)}

Quality Range: {df['quality'].min()}-{df['quality'].max()}
Most Common: {df['quality'].mode()[0]}

Prediction Accuracy: {accuracy*100:.1f}%"""
        
        plt.text(0.1, 0.5, performance_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # 6. Feature Correlations with Quality
        plt.subplot(2, 3, 6)
        correlations = df.corr()['quality'].drop('quality').sort_values(key=abs, ascending=False).head(8)
        plt.barh(range(len(correlations)), correlations.values, color='orange')
        plt.yticks(range(len(correlations)), correlations.index)
        plt.xlabel('Correlation with Quality')
        plt.title('Feature Correlations')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('wine_quality_prediction_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nDashboard saved as: wine_quality_prediction_dashboard.png")
        
        return True
        
    except FileNotFoundError:
        print("Error: WineQT.csv file not found in 'archive' folder.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    print("WINE QUALITY PREDICTION SYSTEM")
    print("=" * 50)
    success = create_wine_quality_dashboard()
    if success:
        print("=" * 50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
    else:
        print("ANALYSIS FAILED!")