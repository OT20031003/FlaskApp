import google.generativeai as genai
import os

# --- APIキーの設定 ---
# 1. 環境変数から読み込む（推奨）
#    事前にターミナルで export GEMINI_API_KEY='YOUR_API_KEY' を実行
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("エラー: GEMINI_API_KEY 環境変数が設定されていません。")
    print("コード内の 'YOUR_API_KEY_HERE' を置き換えてください。\n")
    # 2. コードに直接記述する（非推奨：セキュリティリスクあり）
    # API_KEY = 'YOUR_API_KEY_HERE'
    # if API_KEY == 'YOUR_API_KEY_HERE':
    #     print("--- !!! 警告 !!! ---")
    #     print("コードにAPIキーを直接記述しています。")
    #     print("セキュリティのため、環境変数を使用することを強く推奨します。")
    #     print("----------------------\n")
    #     # 実行を停止する場合は以下のコメントを解除
    #     # exit() 
    # genai.configure(api_key=API_KEY)


print("🤖 Gemini APIで利用可能なモデル一覧\n")

try:
    # モデルの一覧を取得
    models = genai.list_models()

    if not models:
        print("モデルが取得できませんでした。APIキーが正しいか確認してください。")

    # テキスト生成（generateContent）が可能なモデルをフィルタリング
    generative_models = [
        m for m in models 
        if 'generateContent' in m.supported_generation_methods
    ]

    print("--- テキスト生成 (generateContent) 対応モデル ---")
    for m in generative_models:
        print(f"モデル名: {m.name}")
        print(f"  説明: {m.description}")
        # print(f"  サポートメソッド: {m.supported_generation_methods}") # 詳細表示用
        print("-" * 20)

    # (参考) テキスト生成以外（埋め込み等）のモデル
    other_models = [
        m for m in models 
        if 'generateContent' not in m.supported_generation_methods
    ]
    
    if other_models:
        print("\n--- その他（埋め込み等）のモデル ---")
        for m in other_models:
            print(f"モデル名: {m.name}")
            # print(f"  サポートメソッド: {m.supported_generation_methods}") # 詳細表示用
            print("-" * 20)


except Exception as e:
    print(f"モデルの取得中にエラーが発生しました: {e}")
    print("APIキーが正しく設定されているか、ネットワーク接続を確認してください。")