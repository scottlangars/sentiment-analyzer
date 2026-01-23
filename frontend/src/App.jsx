import React, { useState, useEffect } from 'react';
import { Upload, BarChart3, PieChart, Download, Settings, TrendingUp, AlertCircle, FileText, CheckCircle, XCircle, Loader, LogOut, History, User, Lock, Mail } from 'lucide-react';
import { BarChart, Bar, PieChart as RPieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const SentimentDashboard = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [showLogin, setShowLogin] = useState(true);
  const [loginData, setLoginData] = useState({ email: '', password: '' });
  const [signupData, setSignupData] = useState({ name: '', email: '', password: '' });
  
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [translateEnabled, setTranslateEnabled] = useState(false);
  const [filterSentiments, setFilterSentiments] = useState(['POSITIVE', 'NEUTRAL', 'NEGATIVE']);
  const [error, setError] = useState(null);
  const [fileHistory, setFileHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  // Use relative API URL for deployment
  const API_URL = '/api';

  // Load user data from memory
  useEffect(() => {
    const savedUser = localStorage.getItem('currentUser');
    if (savedUser) {
      const user = JSON.parse(savedUser);
      setCurrentUser(user);
      setIsAuthenticated(true);
      loadFileHistory(user.email);
    }
  }, []);

  const loadFileHistory = (email) => {
    const history = JSON.parse(localStorage.getItem(`history_${email}`) || '[]');
    setFileHistory(history);
  };

  const handleSignup = (e) => {
    e.preventDefault();
    const users = JSON.parse(localStorage.getItem('sentimentUsers') || '[]');
    
    if (users.find(u => u.email === signupData.email)) {
      setError('Email already exists');
      return;
    }

    const newUser = {
      id: Date.now(),
      name: signupData.name,
      email: signupData.email,
      password: signupData.password,
      createdAt: new Date().toISOString()
    };

    users.push(newUser);
    localStorage.setItem('sentimentUsers', JSON.stringify(users));
    setError(null);
    alert('Account created successfully! Please login.');
    setShowLogin(true);
  };

  const handleLogin = (e) => {
    e.preventDefault();
    const users = JSON.parse(localStorage.getItem('sentimentUsers') || '[]');
    const user = users.find(u => u.email === loginData.email && u.password === loginData.password);

    if (user) {
      setCurrentUser(user);
      setIsAuthenticated(true);
      localStorage.setItem('currentUser', JSON.stringify(user));
      loadFileHistory(user.email);
      setError(null);
    } else {
      setError('Invalid email or password');
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setCurrentUser(null);
    localStorage.removeItem('currentUser');
    setResults(null);
    setFile(null);
  };

  const saveToHistory = (fileInfo) => {
    const historyKey = `history_${currentUser.email}`;
    const history = JSON.parse(localStorage.getItem(historyKey) || '[]');
    
    const newEntry = {
      id: Date.now(),
      fileName: fileInfo.fileName,
      uploadedAt: new Date().toISOString(),
      totalRecords: fileInfo.totalRecords,
      sentiments: fileInfo.sentiments,
      translated: fileInfo.translated
    };

    history.unshift(newEntry);
    if (history.length > 20) history.pop();
    
    localStorage.setItem(historyKey, JSON.stringify(history));
    setFileHistory(history);
  };

  const analyzeFile = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('translate', translateEnabled);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Analysis failed');
      }

      const data = await response.json();
      setResults(data);
      
      saveToHistory({
        fileName: file.name,
        totalRecords: data.total,
        sentiments: data.sentiments,
        translated: translateEnabled
      });
      
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to analyze file. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      if (uploadedFile.type === 'text/csv' || uploadedFile.name.endsWith('.csv')) {
        setFile(uploadedFile);
        setResults(null);
        setError(null);
      } else {
        setError('Please upload a CSV file');
      }
    }
  };

  const downloadResults = (type) => {
    if (!results) return;

    let csvContent = '';
    const BOM = '\uFEFF';

    if (type === 'full') {
      csvContent = BOM + 'Text,Sentiment,Confidence\n';
      results.samples.forEach(sample => {
        const text = sample.text.replace(/"/g, '""');
        csvContent += `"${text}",${sample.sentiment},${sample.confidence}\n`;
      });
    } else {
      csvContent = BOM + 'Sentiment,Count,Percentage,Avg Confidence\n';
      Object.entries(results.sentiments).forEach(([sentiment, data]) => {
        csvContent += `${sentiment},${data.count},${data.percentage}%,${data.avgConfidence}\n`;
      });
    }

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sentiment_analysis_${type}_${new Date().getTime()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const COLORS = {
    POSITIVE: '#2ecc71',
    NEUTRAL: '#f39c12',
    NEGATIVE: '#e74c3c'
  };

  const chartData = results ? [
    { name: 'POSITIVE', value: results.sentiments.POSITIVE.count, confidence: results.sentiments.POSITIVE.avgConfidence },
    { name: 'NEUTRAL', value: results.sentiments.NEUTRAL.count, confidence: results.sentiments.NEUTRAL.avgConfidence },
    { name: 'NEGATIVE', value: results.sentiments.NEGATIVE.count, confidence: results.sentiments.NEGATIVE.avgConfidence }
  ] : [];

  const filteredSamples = results?.samples.filter(s => filterSentiments.includes(s.sentiment)) || [];

  const toggleFilter = (sentiment) => {
    setFilterSentiments(prev =>
      prev.includes(sentiment)
        ? prev.filter(s => s !== sentiment)
        : [...prev, sentiment]
    );
  };

  // Login/Signup UI
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">💬 Sentiment AI</h1>
            <p className="text-gray-600">Advanced Customer Feedback Analysis</p>
          </div>

          {showLogin ? (
            <form onSubmit={handleLogin} className="space-y-4">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Login</h2>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Mail className="w-4 h-4 inline mr-2" />
                  Email
                </label>
                <input
                  type="email"
                  value={loginData.email}
                  onChange={(e) => setLoginData({...loginData, email: e.target.value})}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Lock className="w-4 h-4 inline mr-2" />
                  Password
                </label>
                <input
                  type="password"
                  value={loginData.password}
                  onChange={(e) => setLoginData({...loginData, password: e.target.value})}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              {error && (
                <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-600" />
                  <p className="text-red-700 text-sm">{error}</p>
                </div>
              )}

              <button
                type="submit"
                className="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all"
              >
                Login
              </button>

              <p className="text-center text-sm text-gray-600">
                Don't have an account?{' '}
                <button
                  type="button"
                  onClick={() => {setShowLogin(false); setError(null);}}
                  className="text-blue-600 font-semibold hover:underline"
                >
                  Sign Up
                </button>
              </p>
            </form>
          ) : (
            <form onSubmit={handleSignup} className="space-y-4">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Sign Up</h2>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <User className="w-4 h-4 inline mr-2" />
                  Full Name
                </label>
                <input
                  type="text"
                  value={signupData.name}
                  onChange={(e) => setSignupData({...signupData, name: e.target.value})}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Mail className="w-4 h-4 inline mr-2" />
                  Email
                </label>
                <input
                  type="email"
                  value={signupData.email}
                  onChange={(e) => setSignupData({...signupData, email: e.target.value})}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Lock className="w-4 h-4 inline mr-2" />
                  Password
                </label>
                <input
                  type="password"
                  value={signupData.password}
                  onChange={(e) => setSignupData({...signupData, password: e.target.value})}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                  minLength={6}
                />
              </div>

              {error && (
                <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-600" />
                  <p className="text-red-700 text-sm">{error}</p>
                </div>
              )}

              <button
                type="submit"
                className="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all"
              >
                Create Account
              </button>

              <p className="text-center text-sm text-gray-600">
                Already have an account?{' '}
                <button
                  type="button"
                  onClick={() => {setShowLogin(true); setError(null);}}
                  className="text-blue-600 font-semibold hover:underline"
                >
                  Login
                </button>
              </p>
            </form>
          )}
        </div>
      </div>
    );
  }

  // Main Dashboard
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                💬 AI Sentiment Analyzer
              </h1>
              <p className="text-gray-600 mt-2">Welcome, {currentUser.name}!</p>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="flex items-center gap-2 px-4 py-2 bg-purple-50 rounded-lg hover:bg-purple-100 transition-colors"
              >
                <History className="w-5 h-5 text-purple-600" />
                <span className="text-sm font-medium text-purple-700">History</span>
              </button>
              <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-lg">
                <Settings className="w-5 h-5 text-blue-600" />
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={translateEnabled}
                    onChange={(e) => setTranslateEnabled(e.target.checked)}
                    className="w-4 h-4 text-blue-600 rounded"
                  />
                  <span className="text-sm font-medium text-gray-700">Translation</span>
                </label>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 bg-red-50 rounded-lg hover:bg-red-100 transition-colors"
              >
                <LogOut className="w-5 h-5 text-red-600" />
                <span className="text-sm font-medium text-red-700">Logout</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* File History Sidebar */}
        {showHistory && (
          <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-200">
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <History className="w-6 h-6 text-purple-600" />
              Analysis History
            </h3>
            {fileHistory.length === 0 ? (
              <p className="text-gray-500 text-center py-4">No analysis history yet</p>
            ) : (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {fileHistory.map((entry) => (
                  <div key={entry.id} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-semibold text-gray-800">{entry.fileName}</p>
                      <span className="text-xs text-gray-500">
                        {new Date(entry.uploadedAt).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex gap-4 text-sm">
                      <span className="text-gray-600">Records: {entry.totalRecords}</span>
                      <span className="text-green-600">+ {entry.sentiments.POSITIVE.count}</span>
                      <span className="text-yellow-600">○ {entry.sentiments.NEUTRAL.count}</span>
                      <span className="text-red-600">- {entry.sentiments.NEGATIVE.count}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Upload Section */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8 border border-gray-200">
          <div className="flex items-center gap-4 mb-6">
            <Upload className="w-8 h-8 text-blue-600" />
            <div>
              <h2 className="text-2xl font-bold text-gray-800">Upload CSV File</h2>
              <p className="text-gray-600">Upload your customer feedback data for analysis</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <p className="text-lg font-medium text-gray-700 mb-2">
                  {file ? file.name : 'Click to upload CSV file'}
                </p>
                <p className="text-sm text-gray-500">
                  {file ? `${(file.size / 1024).toFixed(2)} KB` : 'Supports: .csv files'}
                </p>
              </label>
            </div>

            {error && (
              <div className="flex items-center gap-2 p-4 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-600" />
                <p className="text-red-700">{error}</p>
              </div>
            )}

            <button
              onClick={analyzeFile}
              disabled={!file || analyzing}
              className="w-full py-4 px-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 flex items-center justify-center gap-2"
            >
              {analyzing ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <TrendingUp className="w-5 h-5" />
                  Run Sentiment Analysis
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results Section */}
        {results && (
          <div className="space-y-8">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
                <div className="flex items-center justify-between mb-2">
                  <FileText className="w-8 h-8 text-blue-600" />
                </div>
                <p className="text-gray-600 text-sm mb-1">Total Reviews</p>
                <p className="text-3xl font-bold text-gray-800">{results.total.toLocaleString()}</p>
              </div>

              <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl shadow-lg p-6 border border-green-200">
                <div className="flex items-center justify-between mb-2">
                  <CheckCircle className="w-8 h-8 text-green-600" />
                </div>
                <p className="text-green-800 text-sm mb-1">Positive</p>
                <p className="text-3xl font-bold text-green-900">
                  {results.sentiments.POSITIVE.count}
                  <span className="text-lg ml-2">({results.sentiments.POSITIVE.percentage}%)</span>
                </p>
              </div>

              <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 rounded-xl shadow-lg p-6 border border-yellow-200">
                <div className="flex items-center justify-between mb-2">
                  <AlertCircle className="w-8 h-8 text-yellow-600" />
                </div>
                <p className="text-yellow-800 text-sm mb-1">Neutral</p>
                <p className="text-3xl font-bold text-yellow-900">
                  {results.sentiments.NEUTRAL.count}
                  <span className="text-lg ml-2">({results.sentiments.NEUTRAL.percentage}%)</span>
                </p>
              </div>

              <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-xl shadow-lg p-6 border border-red-200">
                <div className="flex items-center justify-between mb-2">
                  <XCircle className="w-8 h-8 text-red-600" />
                </div>
                <p className="text-red-800 text-sm mb-1">Negative</p>
                <p className="text-3xl font-bold text-red-900">
                  {results.sentiments.NEGATIVE.count}
                  <span className="text-lg ml-2">({results.sentiments.NEGATIVE.percentage}%)</span>
                </p>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <PieChart className="w-6 h-6 text-blue-600" />
                  Sentiment Distribution
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <RPieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[entry.name]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </RPieChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <BarChart3 className="w-6 h-6 text-purple-600" />
                  Average Confidence Scores
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Bar dataKey="confidence" fill="#8884d8">
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[entry.name]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Data Preview */}
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                  <FileText className="w-6 h-6 text-blue-600" />
                  Data Preview
                </h3>
                <div className="flex gap-2">
                  {['POSITIVE', 'NEUTRAL', 'NEGATIVE'].map(sentiment => (
                    <button
                      key={sentiment}
                      onClick={() => toggleFilter(sentiment)}
                      className={`px-4 py-2 rounded-lg font-medium transition-all ${
                        filterSentiments.includes(sentiment)
                          ? 'text-white shadow-md'
                          : 'bg-gray-100 text-gray-600'
                      }`}
                      style={{
                        backgroundColor: filterSentiments.includes(sentiment) ? COLORS[sentiment] : undefined
                      }}
                    >
                      {sentiment}
                    </button>
                  ))}
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Text</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sentiment</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {filteredSamples.slice(0, 10).map((sample, idx) => (
                      <tr key={idx} className="hover:bg-gray-50">
                        <td className="px-6 py-4 text-sm text-gray-900">{sample.text}</td>
                        <td className="px-6 py-4">
                          <span
                            className="px-3 py-1 rounded-full text-xs font-semibold text-white"
                            style={{ backgroundColor: COLORS[sample.sentiment] }}
                          >
                            {sample.sentiment}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900">{sample.confidence.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Download Section */}
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Download className="w-6 h-6 text-blue-600" />
                Download Results
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <button
                  onClick={() => downloadResults('full')}
                  className="py-3 px-6 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-all flex items-center justify-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  Download Full Results (CSV)
                </button>
                <button
                  onClick={() => downloadResults('summary')}
                  className="py-3 px-6 bg-purple-600 text-white font-semibold rounded-lg hover:bg-purple-700 transition-all flex items-center justify-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  Download Summary (CSV)
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Instructions */}
        {!results && !analyzing && (
          <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
            <h3 className="text-2xl font-bold text-gray-800 mb-6">📋 How to Use</h3>
            <div className="space-y-4 text-gray-700">
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">1</span>
                <div>
                  <p className="font-semibold">Upload CSV File</p>
                  <p className="text-sm text-gray-600">Click the upload area and select your CSV file with text data</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">2</span>
                <div>
                  <p className="font-semibold">Configure Settings</p>
                  <p className="text-sm text-gray-600">Enable translation if your data contains non-English text</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">3</span>
                <div>
                  <p className="font-semibold">Run Analysis</p>
                  <p className="text-sm text-gray-600">Click "Run Sentiment Analysis" and wait for the results</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">4</span>
                <div>
                  <p className="font-semibold">View & Download</p>
                  <p className="text-sm text-gray-600">Explore visualizations and download results</p>
                </div>
              </div>
            </div>

            <div className="mt-8 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h4 className="font-semibold text-blue-900 mb-2">📊 CSV Format Requirements:</h4>
              <p className="text-sm text-blue-800 mb-2">Your CSV should have at least one column with text data:</p>
              <code className="text-sm bg-blue-100 px-2 py-1 rounded">text, review, comment, feedback, message, content</code>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SentimentDashboard;