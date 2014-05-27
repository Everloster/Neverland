package org.jabe.neverland.plugin;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import android.content.Context;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.view.ContextThemeWrapper;

public class PluginContextWrapper extends ContextThemeWrapper {

	private AssetManager mAssetManager = null;
	private Resources mResources = null;
	private Resources.Theme mTheme = null;
	private ClassLoader mClassLoader;

	public PluginContextWrapper(Context context, int themeres, String apkPath,
			ClassLoader classLoader) {
		super(context, themeres);
		this.mClassLoader = classLoader;
		this.mAssetManager = initAssetManager(apkPath);
		this.mResources = initResources(context, mAssetManager);
		this.mTheme = initTheme(mResources);
	}

	private AssetManager initAssetManager(String apkPath) {
		AssetManager assetManager = null;
		try {
			assetManager = (AssetManager) AssetManager.class.newInstance();
			Method method = AssetManager.class.getDeclaredMethod("addAssetPath",
					new Class[] { String.class });
			method.invoke(assetManager, new Object[] { apkPath });
		} catch (InstantiationException e) {
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			e.printStackTrace();
		} catch (NoSuchMethodException e) {
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			e.printStackTrace();
		}
		return assetManager;
	}

	private Resources.Theme initTheme(Resources resources) {
		Resources.Theme localTheme = resources.newTheme();
		int themeId = initThemeId("com.android.internal.R.style.Theme");
		localTheme.applyStyle(themeId, true);
		return localTheme;
	}

	private Resources initResources(Context context, AssetManager assetManager) {
		return new Resources(assetManager, context.getResources().getDisplayMetrics(), context
				.getResources().getConfiguration());
	}

	private int initThemeId(String paramString) {
		int i = -1;
		try {
			String str1 = paramString.substring(0, 2 + paramString.indexOf(".R."));
			int j = paramString.lastIndexOf(".");
			String str2 = paramString.substring(j + 1, paramString.length());
			String str3 = paramString.substring(0, j);
			String str4 = str3.substring(1 + str3.lastIndexOf("."), str3.length());
			int k = Class.forName(str1 + "$" + str4).getDeclaredField(str2).getInt(null);
			i = k;
			return i;
		} catch (Throwable localThrowable) {
			while (true)
				localThrowable.printStackTrace();
		}
	}

	public AssetManager getAssets() {
		return this.mAssetManager;
	}

	public ClassLoader getClassLoader() {
		if (this.mClassLoader != null) {
			return mClassLoader;
		} else {
			return super.getClassLoader();
		}
	}

	public Resources getResources() {
		return this.mResources;
	}

	public Resources.Theme getTheme() {
		return this.mTheme;
	}
	
}
