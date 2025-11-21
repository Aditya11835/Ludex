from flask import Flask, redirect, request, session, url_for
import requests, re, os
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")  # any random value

STEAM_OPENID_URL = "https://steamcommunity.com/openid/login"

# path to save SteamIDs
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
SEED_FILE = BASE_DIR / "steam_data" / "seed_steamids.txt"
SEED_FILE.parent.mkdir(parents=True, exist_ok=True)


@app.route("/")
def home():
    return """
    <h1>Steam CF Recommender</h1>
    <a href='/login/steam'>Sign in with Steam</a>
    """


@app.route("/login/steam")
def login_steam():
    # Where Steam should return after login
    return_to = url_for('steam_return', _external=True)
    realm = request.url_root

    params = {
        'openid.ns': 'http://specs.openid.net/auth/2.0',
        'openid.mode': 'checkid_setup',
        'openid.return_to': return_to,
        'openid.realm': realm,
        'openid.identity': 'http://specs.openid.net/auth/2.0/identifier_select',
        'openid.claimed_id': 'http://specs.openid.net/auth/2.0/identifier_select',
    }

    # Redirect user to Steam login page
    req = requests.Request('GET', STEAM_OPENID_URL, params=params).prepare()
    return redirect(req.url)


@app.route("/auth/steam/return")
def steam_return():
    # Validate with Steam
    openid_params = {k: v for k, v in request.args.items() if k.startswith('openid.')}
    openid_params['openid.mode'] = 'check_authentication'

    resp = requests.post(STEAM_OPENID_URL, data=openid_params)

    if "is_valid:true" not in resp.text:
        return "Steam login failed", 400

    # Extract SteamID64 from claimed_id
    claimed = request.args.get("openid.claimed_id")
    match = re.search(r"/id/(\d+)$", claimed)

    if not match:
        return "Error extracting SteamID", 400

    steamid = match.group(1)
    session['steamid'] = steamid

    # Save the SteamID to seed file
    existing = []
    if SEED_FILE.exists():
        existing = [l.strip() for l in SEED_FILE.read_text().splitlines()]

    if steamid not in existing:
        existing.append(steamid)
        SEED_FILE.write_text("\n".join(existing))

    return f"""
        <h2>Login successful!</h2>
        <p>Your SteamID: {steamid}</p>
        <p>Saved to seed_steamids.txt</p>
    """


if __name__ == "__main__":
    app.run(port=5000, debug=True)
