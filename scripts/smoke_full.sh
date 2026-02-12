# ... (Keep existing script) ...
# Appending worker smoke test

echo "Building Worker Image..."
# Override entrypoint to run worker
docker build --target runtime -t market-dashboard:worker .

echo "Running Worker Smoke Test..."
docker run --rm market-dashboard:worker python -m src.run_backtest_cli --sample --days 50

if [ $? -eq 0 ]; then
    echo "✅ Worker smoke test passed!"
else
    echo "❌ Worker smoke test failed!"
    exit 1
fi
