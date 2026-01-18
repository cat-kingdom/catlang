# Konfigurasi MCP Cursor untuk CatLang

File `mcp.json` ini mengkonfigurasi CatLang MCP server untuk digunakan dengan Cursor IDE.

## ⚠️ Langkah Penting: Ganti API Key

File ini menggunakan **placeholder API key**. Anda **harus mengganti** dengan API key OpenRouter Anda yang sebenarnya.

### Cara Mengganti API Key

1. Buka file `.cursor/mcp.json`
2. Ganti `sk-or-v1-your-openrouter-api-key-here` dengan API key OpenRouter Anda yang sebenarnya
3. Simpan file
4. Restart Cursor IDE

### Contoh Setelah Diganti

```json
{
  "mcpServers": {
    "catlang": {
      "command": "python",
      "args": [
        "/home/muhfajarags/project/catlang/run_mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/home/muhfajarags/project/catlang",
        "LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-or-v1-abc123xyz456...",
        "OPENROUTER_MODEL": "x-ai/grok-beta"
      }
    }
  }
}
```

## Konfigurasi Saat Ini

- **Provider**: OpenRouter
- **Model**: `x-ai/grok-beta` (Grok Beta via OpenRouter)
- **Path**: Absolute path ke `run_mcp_server.py`

## Mengubah Model

Jika ingin menggunakan model lain, ubah `OPENROUTER_MODEL`:

- `x-ai/grok-beta` - Grok Beta
- `x-ai/grok-2` - Grok 2
- `openai/gpt-4o` - GPT-4o
- `openai/gpt-4o-mini` - GPT-4o Mini (lebih murah)
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet

Lihat semua model di: https://openrouter.ai/models

## Verifikasi Setup

Setelah mengganti API key dan restart Cursor:

1. **Cek status server:**
   ```bash
   cursor-agent mcp list
   ```

2. **List tools yang tersedia:**
   ```bash
   cursor-agent mcp list-tools catlang
   ```

3. **Expected output (6 tools):**
   - analyze_n8n_workflow
   - extract_custom_logic
   - generate_langgraph_implementation
   - validate_implementation
   - list_guides
   - query_guide

## Troubleshooting

### Server tidak muncul di Cursor
- Pastikan JSON syntax valid (tidak ada typo)
- Restart Cursor setelah perubahan
- Cek path ke `run_mcp_server.py` benar

### API key tidak valid
- Pastikan API key sudah di-copy dengan benar (tidak ada spasi)
- Format OpenRouter: `sk-or-v1-...`
- Test API key di https://openrouter.ai/keys

### Tools tidak tersedia
- Pastikan server status "Connected" di Cursor Settings
- Gunakan Agent mode di Composer
- Cek log server untuk error
