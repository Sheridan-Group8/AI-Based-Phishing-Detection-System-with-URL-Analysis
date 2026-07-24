'use strict';

const fs = require('fs');
const path = require('path');

const PROVIDER_IDS = Object.freeze([
    'virustotal',
    'google_safe_browsing',
    'urlscan',
    'abuseipdb',
]);

class ProviderSettingsStore {
    constructor({ userDataDir, safeStorage, platform = process.platform }) {
        if (!userDataDir || typeof userDataDir !== 'string') {
            throw new TypeError('userDataDir is required');
        }
        this.filePath = path.join(userDataDir, 'threat-intelligence-settings.json');
        this.safeStorage = safeStorage;
        this.platform = platform;
        this.data = this._read();
        this.unavailableKeys = new Set();
    }

    _blankData() {
        return { version: 1, providers: {} };
    }

    _read() {
        try {
            const parsed = JSON.parse(fs.readFileSync(this.filePath, 'utf8'));
            if (!parsed || parsed.version !== 1 || typeof parsed.providers !== 'object') {
                return this._blankData();
            }
            const providers = {};
            for (const id of PROVIDER_IDS) {
                const source = parsed.providers[id];
                if (!source || typeof source !== 'object') continue;
                const item = {};
                if (typeof source.enabled === 'boolean') item.enabled = source.enabled;
                if (typeof source.encryptedKey === 'string' && source.encryptedKey) {
                    item.encryptedKey = source.encryptedKey;
                }
                if (Object.keys(item).length) providers[id] = item;
            }
            return { version: 1, providers };
        } catch (error) {
            if (error && error.code !== 'ENOENT') {
                // Do not include file contents or parse details in logs: a damaged
                // encrypted value still belongs to the user's credential data.
                console.warn('Threat-intelligence settings could not be read; using defaults.');
            }
            return this._blankData();
        }
    }

    _write() {
        fs.mkdirSync(path.dirname(this.filePath), { recursive: true, mode: 0o700 });
        const temporary = `${this.filePath}.${process.pid}.tmp`;
        fs.writeFileSync(temporary, JSON.stringify(this.data), {
            encoding: 'utf8',
            mode: 0o600,
            flag: 'w',
        });
        fs.renameSync(temporary, this.filePath);
        try {
            fs.chmodSync(this.filePath, 0o600);
        } catch (_error) {
            // Windows ACLs are inherited from Electron's userData directory.
        }
    }

    storageSecurity() {
        let available = false;
        let backend = this.platform === 'linux' ? 'unknown' : 'os_protected';
        try {
            available = Boolean(
                this.safeStorage
                && typeof this.safeStorage.isEncryptionAvailable === 'function'
                && this.safeStorage.isEncryptionAvailable()
            );
            if (
                this.platform === 'linux'
                && this.safeStorage
                && typeof this.safeStorage.getSelectedStorageBackend === 'function'
            ) {
                backend = this.safeStorage.getSelectedStorageBackend();
            }
        } catch (_error) {
            available = false;
        }
        const secure = available && backend !== 'basic_text';
        return {
            available: secure,
            backend,
            reason: secure
                ? ''
                : (backend === 'basic_text'
                    ? 'No protected desktop keyring is available.'
                    : 'OS credential encryption is unavailable.'),
        };
    }

    publicSettings() {
        const providers = {};
        for (const id of PROVIDER_IDS) {
            const item = this.data.providers[id] || {};
            providers[id] = {
                enabledPreference: typeof item.enabled === 'boolean' ? item.enabled : null,
                userKeyConfigured: Boolean(item.encryptedKey),
                userKeyAvailable: Boolean(item.encryptedKey)
                    && !this.unavailableKeys.has(id),
            };
        }
        return {
            secureStorage: this.storageSecurity(),
            providers,
        };
    }

    backendConfiguration() {
        const providers = {};
        for (const id of PROVIDER_IDS) {
            const item = this.data.providers[id] || {};
            const output = { use_deployer_key: !item.encryptedKey };
            if (typeof item.enabled === 'boolean') output.enabled = item.enabled;
            if (item.encryptedKey) {
                try {
                    output.key = this.safeStorage.decryptString(
                        Buffer.from(item.encryptedKey, 'base64')
                    );
                    output.use_deployer_key = false;
                    this.unavailableKeys.delete(id);
                } catch (_error) {
                    // A key encrypted under a different OS user/keyring cannot be
                    // recovered. Do not expose the ciphertext or failure details.
                    output.use_deployer_key = true;
                    output.key_unavailable = true;
                    this.unavailableKeys.add(id);
                }
            }
            providers[id] = output;
        }
        return { type: 'provider_config', providers };
    }

    update(providerId, changes) {
        if (!PROVIDER_IDS.includes(providerId)) {
            throw new TypeError('Unknown provider');
        }
        if (!changes || typeof changes !== 'object' || Array.isArray(changes)) {
            throw new TypeError('Invalid settings update');
        }

        const item = { ...(this.data.providers[providerId] || {}) };
        if (Object.prototype.hasOwnProperty.call(changes, 'enabled')) {
            if (typeof changes.enabled !== 'boolean') {
                throw new TypeError('enabled must be a boolean');
            }
            item.enabled = changes.enabled;
        }

        if (Object.prototype.hasOwnProperty.call(changes, 'key')) {
            if (typeof changes.key !== 'string') {
                throw new TypeError('key must be a string');
            }
            const key = changes.key.trim();
            if (!key || key.length > 4096 || !/^[\x21-\x7e]+$/.test(key)) {
                throw new TypeError('API key is empty or invalid');
            }
            const security = this.storageSecurity();
            if (!security.available) {
                const error = new Error(security.reason);
                error.code = 'SECURE_STORAGE_UNAVAILABLE';
                throw error;
            }
            item.encryptedKey = this.safeStorage.encryptString(key).toString('base64');
            this.unavailableKeys.delete(providerId);
        }

        this.data.providers[providerId] = item;
        this._write();
        return this.publicSettings();
    }

    clearUserKey(providerId) {
        if (!PROVIDER_IDS.includes(providerId)) {
            throw new TypeError('Unknown provider');
        }
        const item = { ...(this.data.providers[providerId] || {}) };
        delete item.encryptedKey;
        this.unavailableKeys.delete(providerId);
        if (Object.keys(item).length) this.data.providers[providerId] = item;
        else delete this.data.providers[providerId];
        this._write();
        return this.publicSettings();
    }
}

module.exports = { PROVIDER_IDS, ProviderSettingsStore };
