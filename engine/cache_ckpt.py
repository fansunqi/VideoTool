import sqlite3
con = sqlite3.connect('/home/fsq/.cache/octotools/cache_openai_gpt-4o.db/cache.db')
con.execute('PRAGMA wal_checkpoint(FULL);')
con.close()
