#!/usr/bin/env python3
"""
Autonomous AI Investment Engine
Self-learning system that continuously analyzes markets and makes decisions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sqlite3
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import asyncio
import aiohttp
from dataclasses import dataclass
import pickle
import time

@dataclass
class MarketSignal:
    """Structured market signal data"""
    ticker: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float   # 0-1
    confidence: float # 0-1
    reasoning: str
    factors: Dict
    timestamp: datetime

class AutonomousAIEngine:
    """Self-learning AI investment system"""
    
    def __init__(self, db_path: str = "data/unified_stock_intelligence.db"):
        self.db_path = db_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.performance_history = []
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ai_engine.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def get_full_stock_universe(self) -> List[str]:
        """Get comprehensive list of tradeable stocks"""
        
        # Start with major indices
        base_tickers = []
        
        # S&P 500
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            tables = pd.read_html(sp500_url)
            sp500_tickers = tables[0]['Symbol'].tolist()
            base_tickers.extend(sp500_tickers)
            self.logger.info(f"Added {len(sp500_tickers)} S&P 500 stocks")
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500: {e}")
        
        # NASDAQ 100
        nasdaq100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA',
            'AVGO', 'COST', 'NFLX', 'TMUS', 'CSCO', 'ADBE', 'PEP', 'LIN',
            'TXN', 'QCOM', 'CMCSA', 'INTU', 'AMGN', 'HON', 'AMD', 'AMAT'
            # Add more as needed
        ]
        base_tickers.extend(nasdaq100_tickers)
        
        # Add popular ETFs and growth stocks
        popular_additions = [
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'VTI', 'ARKK', 'SOXL', 'TQQQ',
            'PLTR', 'SNOW', 'CRWD', 'NET', 'DDOG', 'ZM', 'SHOP', 'SQ', 'ROKU',
            'COIN', 'HOOD', 'RIVN', 'LCID', 'F', 'GM', 'NIO', 'XPEV', 'LI'
        ]
        base_tickers.extend(popular_additions)
        
        # Remove duplicates and invalid tickers
        unique_tickers = list(set(base_tickers))
        
        # Filter out invalid tickers
        valid_tickers = []
        for ticker in unique_tickers[:200]:  # Start with 200 for testing
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if info and 'regularMarketPrice' in info:
                    valid_tickers.append(ticker)
            except:
                continue
        
        self.logger.info(f"Final universe: {len(valid_tickers)} valid tickers")
        return valid_tickers

    def extract_advanced_features(self, ticker: str) -> Dict:
        """Extract comprehensive features for AI analysis"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get multiple timeframes of data
            hist_1y = stock.history(period="1y")
            hist_3m = stock.history(period="3mo")
            hist_1m = stock.history(period="1mo")
            
            if hist_1y.empty:
                return None
            
            features = {}
            
            # Price-based features
            current_price = hist_1y['Close'].iloc[-1]
            features['current_price'] = current_price
            features['price_change_1d'] = (hist_1y['Close'].iloc[-1] / hist_1y['Close'].iloc[-2] - 1)
            features['price_change_1w'] = (hist_1y['Close'].iloc[-1] / hist_1y['Close'].iloc[-5] - 1) if len(hist_1y) >= 5 else 0
            features['price_change_1m'] = (hist_1y['Close'].iloc[-1] / hist_1y['Close'].iloc[-21] - 1) if len(hist_1y) >= 21 else 0
            features['price_change_3m'] = (hist_1y['Close'].iloc[-1] / hist_1y['Close'].iloc[-63] - 1) if len(hist_1y) >= 63 else 0
            
            # Technical indicators
            features.update(self.calculate_technical_features(hist_1y))
            
            # Volume features
            features['avg_volume'] = hist_1y['Volume'].mean()
            features['volume_trend'] = hist_1y['Volume'].rolling(20).mean().iloc[-1] / hist_1y['Volume'].rolling(50).mean().iloc[-1] if len(hist_1y) >= 50 else 1
            features['volume_spike'] = hist_1y['Volume'].iloc[-1] / hist_1y['Volume'].rolling(20).mean().iloc[-1] if len(hist_1y) >= 20 else 1
            
            # Volatility features
            features['volatility_20d'] = hist_1y['Close'].rolling(20).std().iloc[-1] if len(hist_1y) >= 20 else 0
            features['volatility_60d'] = hist_1y['Close'].rolling(60).std().iloc[-1] if len(hist_1y) >= 60 else 0
            
            # Try to get fundamental data
            try:
                info = stock.info
                features['market_cap'] = info.get('marketCap', 0)
                features['pe_ratio'] = info.get('forwardPE', 0) or info.get('trailingPE', 0) or 0
                features['pb_ratio'] = info.get('priceToBook', 0)
                features['debt_to_equity'] = info.get('debtToEquity', 0)
                features['roe'] = info.get('returnOnEquity', 0)
                features['profit_margin'] = info.get('profitMargins', 0)
                features['revenue_growth'] = info.get('revenueGrowth', 0)
                features['sector'] = info.get('sector', 'Unknown')
            except:
                # Set defaults if fundamental data unavailable
                features.update({
                    'market_cap': 0, 'pe_ratio': 0, 'pb_ratio': 0, 'debt_to_equity': 0,
                    'roe': 0, 'profit_margin': 0, 'revenue_growth': 0, 'sector': 'Unknown'
                })
            
            # Market context features
            try:
                spy = yf.Ticker('SPY').history(period="1y")
                if not spy.empty:
                    spy_return = spy['Close'].iloc[-1] / spy['Close'].iloc[-21] - 1 if len(spy) >= 21 else 0
                    features['market_return_1m'] = spy_return
                    features['beta'] = np.corrcoef(hist_1y['Close'].pct_change().dropna(), 
                                                 spy['Close'].pct_change().dropna())[0,1] if len(hist_1y) == len(spy) else 1
                else:
                    features['market_return_1m'] = 0
                    features['beta'] = 1
            except:
                features['market_return_1m'] = 0
                features['beta'] = 1
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features for {ticker}: {e}")
            return None

    def calculate_technical_features(self, df: pd.DataFrame) -> Dict:
        """Calculate technical analysis features"""
        features = {}
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            if len(df) >= period:
                ma = df['Close'].rolling(period).mean().iloc[-1]
                features[f'sma_{period}'] = ma
                features[f'price_vs_sma_{period}'] = df['Close'].iloc[-1] / ma - 1
            else:
                features[f'sma_{period}'] = df['Close'].iloc[-1]
                features[f'price_vs_sma_{period}'] = 0
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features['rsi'] = rsi.iloc[-1] if not rsi.empty else 50
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd.iloc[-1] if not macd.empty else 0
        features['macd_signal'] = signal.iloc[-1] if not signal.empty else 0
        features['macd_histogram'] = (macd - signal).iloc[-1] if not macd.empty and not signal.empty else 0
        
        # Bollinger Bands
        if len(df) >= 20:
            bb_middle = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            features['bb_position'] = (df['Close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            features['bb_squeeze'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
        else:
            features['bb_position'] = 0.5
            features['bb_squeeze'] = 0
        
        return features

    async def collect_training_data(self, lookback_days: int = 90) -> pd.DataFrame:
        """Collect historical data for training"""
        self.logger.info("Collecting training data...")
        
        universe = await self.get_full_stock_universe()
        training_data = []
        
        for ticker in universe:
            try:
                # Get historical data
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                
                if len(hist) < 100:  # Need sufficient data
                    continue
                
                # Create samples for different time points
                for i in range(50, len(hist) - 20, 10):  # Sample every 10 days
                    # Features from day i
                    sample_hist = hist.iloc[:i+1]
                    features = self.extract_features_from_hist(sample_hist, ticker)
                    
                    if features is None:
                        continue
                    
                    # Future return (target) - 20 days forward
                    future_price = hist['Close'].iloc[i + 20]
                    current_price = hist['Close'].iloc[i]
                    future_return = (future_price / current_price) - 1
                    
                    # Classification target
                    if future_return > 0.1:  # 10%+ gain
                        target = 2  # Strong Buy
                    elif future_return > 0.03:  # 3%+ gain
                        target = 1  # Buy
                    elif future_return > -0.03:  # -3% to +3%
                        target = 0  # Hold
                    else:  # >3% loss
                        target = -1  # Sell
                    
                    features['ticker'] = ticker
                    features['target'] = target
                    features['future_return'] = future_return
                    features['date'] = hist.index[i]
                    
                    training_data.append(features)
                
            except Exception as e:
                self.logger.error(f"Error processing {ticker} for training: {e}")
                continue
        
        df = pd.DataFrame(training_data)
        self.logger.info(f"Collected {len(df)} training samples")
        return df

    def extract_features_from_hist(self, hist: pd.DataFrame, ticker: str) -> Dict:
        """Extract features from historical data slice"""
        if len(hist) < 20:
            return None
        
        features = {}
        
        # Price features
        current_price = hist['Close'].iloc[-1]
        features['current_price'] = current_price
        features['price_change_5d'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-6] - 1) if len(hist) >= 6 else 0
        features['price_change_20d'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1) if len(hist) >= 21 else 0
        
        # Technical features
        features.update(self.calculate_technical_features(hist))
        
        # Volume features
        features['avg_volume'] = hist['Volume'].mean()
        features['volume_trend'] = hist['Volume'].iloc[-5:].mean() / hist['Volume'].iloc[-20:-5].mean() if len(hist) >= 20 else 1
        
        # Volatility
        features['volatility'] = hist['Close'].pct_change().std()
        
        return features

    def train_ai_model(self, training_data: pd.DataFrame):
        """Train the AI investment model"""
        self.logger.info("Training AI model...")
        
        # Prepare features
        feature_cols = [col for col in training_data.columns 
                       if col not in ['ticker', 'target', 'future_return', 'date', 'sector']]
        
        X = training_data[feature_cols].fillna(0)
        y = training_data['target']
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        self.feature_columns = feature_cols
        
        self.logger.info(f"Model trained - Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'training_date': datetime.now()
        }
        
        with open('models/ai_investment_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

    def predict_investment_decision(self, ticker: str) -> MarketSignal:
        """Make investment prediction for a ticker"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Extract current features
        features = self.extract_advanced_features(ticker)
        if features is None:
            return MarketSignal(
                ticker=ticker,
                signal_type='HOLD',
                strength=0,
                confidence=0,
                reasoning="Unable to extract features",
                factors={},
                timestamp=datetime.now()
            )
        
        # Prepare features for prediction
        feature_values = []
        for col in self.feature_columns:
            if col in features:
                val = features[col]
                if isinstance(val, str):
                    val = 0  # Handle categorical features
                elif np.isnan(val) or np.isinf(val):
                    val = 0
                feature_values.append(val)
            else:
                feature_values.append(0)
        
        # Scale and predict
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Convert prediction to signal
        signal_map = {2: 'STRONG_BUY', 1: 'BUY', 0: 'HOLD', -1: 'SELL'}
        signal_type = signal_map.get(prediction, 'HOLD')
        
        # Get confidence (max probability)
        confidence = max(probabilities)
        
        # Get feature importance for explanation
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate reasoning
        reasoning = f"AI predicts {signal_type} with {confidence:.1%} confidence. "
        reasoning += f"Key factors: {', '.join([f'{factor}' for factor, _ in top_factors[:3]])}"
        
        return MarketSignal(
            ticker=ticker,
            signal_type=signal_type,
            strength=confidence,
            confidence=confidence,
            reasoning=reasoning,
            factors=dict(top_factors),
            timestamp=datetime.now()
        )

    async def analyze_full_universe(self) -> List[MarketSignal]:
        """Analyze entire stock universe"""
        self.logger.info("Analyzing full universe...")
        
        universe = await self.get_full_stock_universe()
        signals = []
        
        for ticker in universe:
            try:
                signal = self.predict_investment_decision(ticker)
                signals.append(signal)
                
                # Store in database
                self.store_signal_to_db(signal)
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error analyzing {ticker}: {e}")
                continue
        
        # Sort by signal strength
        signals.sort(key=lambda x: x.strength, reverse=True)
        
        self.logger.info(f"Analysis complete: {len(signals)} signals generated")
        return signals

    def store_signal_to_db(self, signal: MarketSignal):
        """Store AI signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS AI_Signals (
                signal_id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength REAL NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT,
                factors JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            INSERT INTO AI_Signals 
            (ticker, signal_type, strength, confidence, reasoning, factors)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            signal.ticker,
            signal.signal_type,
            signal.strength,
            signal.confidence,
            signal.reasoning,
            json.dumps(signal.factors)
        ))
        
        conn.commit()
        conn.close()

    async def continuous_learning_loop(self):
        """Continuous learning and retraining"""
        while True:
            try:
                self.logger.info("Starting continuous learning cycle...")
                
                # Collect new training data
                new_data = await self.collect_training_data()
                
                if len(new_data) > 1000:  # Only retrain if sufficient new data
                    self.train_ai_model(new_data)
                    self.logger.info("Model retrained with new data")
                
                # Analyze full universe
                signals = await self.analyze_full_universe()
                
                # Wait before next cycle (e.g., daily)
                await asyncio.sleep(24 * 60 * 60)  # 24 hours
                
            except Exception as e:
                self.logger.error(f"Error in continuous learning loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    def get_top_signals(self, signal_type: str = None, limit: int = 50) -> pd.DataFrame:
        """Get top AI signals from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT ticker, signal_type, strength, confidence, reasoning, created_at
            FROM AI_Signals
            WHERE DATE(created_at) = DATE('now')
        """
        
        if signal_type:
            query += f" AND signal_type = '{signal_type}'"
        
        query += " ORDER BY strength DESC LIMIT ?"
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df

# Initialize and run the AI engine
async def main():
    """Main execution function"""
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    ai_engine = AutonomousAIEngine()
    
    # Check if model exists
    if not os.path.exists('models/ai_investment_model.pkl'):
        print("Training AI model for the first time...")
        training_data = await ai_engine.collect_training_data()
        ai_engine.train_ai_model(training_data)
    else:
        # Load existing model
        with open('models/ai_investment_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            ai_engine.model = model_data['model']
            ai_engine.scaler = model_data['scaler']
            ai_engine.feature_columns = model_data['feature_columns']
        print("Loaded existing AI model")
    
    # Analyze current universe
    signals = await ai_engine.analyze_full_universe()
    
    print(f"\n🚀 AI Analysis Complete!")
    print(f"📊 Generated {len(signals)} signals")
    
    # Show top buy signals
    buy_signals = [s for s in signals if 'BUY' in s.signal_type]
    print(f"\n🟢 Top Buy Signals:")
    for signal in buy_signals[:10]:
        print(f"  {signal.ticker}: {signal.signal_type} ({signal.confidence:.1%} confidence)")

if __name__ == "__main__":
    asyncio.run(main())