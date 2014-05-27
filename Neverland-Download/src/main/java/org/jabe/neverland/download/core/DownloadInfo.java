package org.jabe.neverland.download.core;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class DownloadInfo {

	public static final DownloadInfo EMPTY;

	// some default param names
	public static final String P_PACKAGENAME = "packageName";
	public static final String P_DOWNLOAD_URI = "downloadUri";
	public static final String P_ICON_URL = "iconUrl";
	public static final String P_DOWNLOAD_NAME = "name";
	public static final String P_SIZE = "size";
	public static final String P_ID = "id";
	
	public static final String P_SECTION_COUNT = "sectionCount";
	public static final String P_WORKER_COUNT = "workerCount";
	
	public DownloadStatus currentDownloadStatus = DownloadStatus.DOWNLOAD_STATUS_PREPARE;
	
	static {
		EMPTY = new DownloadInfo();
		EMPTY.mMap = Collections.unmodifiableMap(EMPTY.mMap);
	}

	public Map<String, Object> mMap = null;

	public DownloadInfo() {
		mMap = new HashMap<String, Object>();
	}

	/**
	 * Inserts all mappings from the given Bundle into this Bundle.
	 * 
	 * @param map
	 *            a Bundle
	 */
	public void putAll(DownloadInfo map) {
		mMap.putAll(map.mMap);
		;
	}

	/**
	 * Inserts a Boolean value into the mapping of this Bundle, replacing any
	 * existing value for the given key. Either key or value may be null.
	 * 
	 * @param key
	 *            a String, or null
	 * @param value
	 *            a Boolean, or null
	 */
	public void putBoolean(String key, boolean value) {
		mMap.put(key, value);
	}

	/**
	 * Inserts a byte value into the mapping of this Bundle, replacing any
	 * existing value for the given key.
	 * 
	 * @param key
	 *            a String, or null
	 * @param value
	 *            a byte
	 */
	public void putByte(String key, byte value) {
		mMap.put(key, value);
	}

	/**
	 * Inserts a char value into the mapping of this Bundle, replacing any
	 * existing value for the given key.
	 * 
	 * @param key
	 *            a String, or null
	 * @param value
	 *            a char, or null
	 */
	public void putChar(String key, char value) {
		mMap.put(key, value);
	}

	/**
	 * Inserts a short value into the mapping of this Bundle, replacing any
	 * existing value for the given key.
	 * 
	 * @param key
	 *            a String, or null
	 * @param value
	 *            a short
	 */
	public void putShort(String key, short value) {
		mMap.put(key, value);
	}

	/**
	 * Inserts an int value into the mapping of this Bundle, replacing any
	 * existing value for the given key.
	 * 
	 * @param key
	 *            a String, or null
	 * @param value
	 *            an int, or null
	 */
	public void putInt(String key, int value) {
		mMap.put(key, value);
	}

	/**
	 * Inserts a long value into the mapping of this Bundle, replacing any
	 * existing value for the given key.
	 * 
	 * @param key
	 *            a String, or null
	 * @param value
	 *            a long
	 */
	public void putLong(String key, long value) {
		mMap.put(key, value);
	}

	/**
	 * Inserts a float value into the mapping of this Bundle, replacing any
	 * existing value for the given key.
	 * 
	 * @param key
	 *            a String, or null
	 * @param value
	 *            a float
	 */
	public void putFloat(String key, float value) {
		mMap.put(key, value);
	}

	/**
	 * Inserts a double value into the mapping of this Bundle, replacing any
	 * existing value for the given key.
	 * 
	 * @param key
	 *            a String, or null
	 * @param value
	 *            a double
	 */
	public void putDouble(String key, double value) {
		mMap.put(key, value);
	}

	/**
	 * Inserts a String value into the mapping of this Bundle, replacing any
	 * existing value for the given key. Either key or value may be null.
	 * 
	 * @param key
	 *            a String, or null
	 * @param value
	 *            a String, or null
	 */
	public void putString(String key, String value) {
		mMap.put(key, value);
	}

	/**
	 * Inserts a CharSequence value into the mapping of this Bundle, replacing
	 * any existing value for the given key. Either key or value may be null.
	 * 
	 * @param key
	 *            a String, or null
	 * @param value
	 *            a CharSequence, or null
	 */
	public void putCharSequence(String key, CharSequence value) {
		mMap.put(key, value);
	}
	
	
	/**
     * Returns the value associated with the given key, or defaultValue if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a boolean value
     */
    public boolean getBoolean(String key, boolean defaultValue) {
        Object o = mMap.get(key);
        if (o == null) {
            return defaultValue;
        }
        try {
            return (Boolean) o;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }

    /**
     * Returns the value associated with the given key, or (byte) 0 if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a byte value
     */
    public byte getByte(String key) {
        return getByte(key, (byte) 0);
    }

    /**
     * Returns the value associated with the given key, or defaultValue if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a byte value
     */
    public Byte getByte(String key, byte defaultValue) {
        Object o = mMap.get(key);
        if (o == null) {
            return defaultValue;
        }
        try {
            return (Byte) o;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }

    /**
     * Returns the value associated with the given key, or false if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a char value
     */
    public char getChar(String key) {
        return getChar(key, (char) 0);
    }

    /**
     * Returns the value associated with the given key, or (char) 0 if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a char value
     */
    public char getChar(String key, char defaultValue) {
        Object o = mMap.get(key);
        if (o == null) {
            return defaultValue;
        }
        try {
            return (Character) o;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }

    /**
     * Returns the value associated with the given key, or (short) 0 if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a short value
     */
    public short getShort(String key) {
        return getShort(key, (short) 0);
    }

    /**
     * Returns the value associated with the given key, or defaultValue if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a short value
     */
    public short getShort(String key, short defaultValue) {
        Object o = mMap.get(key);
        if (o == null) {
            return defaultValue;
        }
        try {
            return (Short) o;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }

    /**
     * Returns the value associated with the given key, or 0 if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return an int value
     */
    public int getInt(String key) {
        return getInt(key, 0);
    }

    /**
     * Returns the value associated with the given key, or defaultValue if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return an int value
     */
    public int getInt(String key, int defaultValue) {
        Object o = mMap.get(key);
        if (o == null) {
            return defaultValue;
        }
        try {
            return (Integer) o;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }

    /**
     * Returns the value associated with the given key, or 0L if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a long value
     */
    public long getLong(String key) {
        return getLong(key, 0L);
    }

    /**
     * Returns the value associated with the given key, or defaultValue if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a long value
     */
    public long getLong(String key, long defaultValue) {
        Object o = mMap.get(key);
        if (o == null) {
            return defaultValue;
        }
        try {
            return (Long) o;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }

    /**
     * Returns the value associated with the given key, or 0.0f if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a float value
     */
    public float getFloat(String key) {
        return getFloat(key, 0.0f);
    }

    /**
     * Returns the value associated with the given key, or defaultValue if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a float value
     */
    public float getFloat(String key, float defaultValue) {
        Object o = mMap.get(key);
        if (o == null) {
            return defaultValue;
        }
        try {
            return (Float) o;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }

    /**
     * Returns the value associated with the given key, or 0.0 if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a double value
     */
    public double getDouble(String key) {
        return getDouble(key, 0.0);
    }

    /**
     * Returns the value associated with the given key, or defaultValue if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String
     * @return a double value
     */
    public double getDouble(String key, double defaultValue) {
        Object o = mMap.get(key);
        if (o == null) {
            return defaultValue;
        }
        try {
            return (Double) o;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }

    /**
     * Returns the value associated with the given key, or null if
     * no mapping of the desired type exists for the given key or a null
     * value is explicitly associated with the key.
     *
     * @param key a String, or null
     * @return a String value, or null
     */
    public String getString(String key) {
        Object o = mMap.get(key);
        if (o == null) {
            return null;
        }
        try {
            return (String) o;
        } catch (ClassCastException e) {
            return null;
        }
    }

    /**
     * Returns the value associated with the given key, or defaultValue if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String, or null
     * @param defaultValue Value to return if key does not exist
     * @return a String value, or null
     */
    public String getString(String key, String defaultValue) {
        Object o = mMap.get(key);
        if (o == null) {
            return defaultValue;
        }
        try {
            return (String) o;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }

    /**
     * Returns the value associated with the given key, or null if
     * no mapping of the desired type exists for the given key or a null
     * value is explicitly associated with the key.
     *
     * @param key a String, or null
     * @return a CharSequence value, or null
     */
    public CharSequence getCharSequence(String key) {
        Object o = mMap.get(key);
        if (o == null) {
            return null;
        }
        try {
            return (CharSequence) o;
        } catch (ClassCastException e) {
            return null;
        }
    }

    /**
     * Returns the value associated with the given key, or defaultValue if
     * no mapping of the desired type exists for the given key.
     *
     * @param key a String, or null
     * @param defaultValue Value to return if key does not exist
     * @return a CharSequence value, or null
     */
    public CharSequence getCharSequence(String key, CharSequence defaultValue) {
        Object o = mMap.get(key);
        if (o == null) {
            return defaultValue;
        }
        try {
            return (CharSequence) o;
        } catch (ClassCastException e) {
            return defaultValue;
        }
    }
    
    public String getmDownloadUrl() {
		return getString(P_DOWNLOAD_URI, "");
	}

	public void setmDownloadUrl(String mDownloadUrl) {
		putString(P_DOWNLOAD_URI, mDownloadUrl);
	}

	public String getmPackageName() {
		return getString(P_PACKAGENAME, "");
	}

	public void setmPackageName(String mPackageName) {
		putString(P_PACKAGENAME, mPackageName);
	}

	public String getmName() {
		return getString(P_DOWNLOAD_NAME, "");
	}

	public void setmName(String mName) {
		putString(P_DOWNLOAD_NAME, mName);
	}

	public String getmIconUrl() {
		return getString(P_ICON_URL, "");
	}

	public void setmIconUrl(String mIconUrl) {
		putString(P_ICON_URL, mIconUrl);
	}

	public String getmId() {
		return getString(P_ID, "");
	}

	public void setmId(String mId) {
		putString(P_ID, mId);
	}

}
