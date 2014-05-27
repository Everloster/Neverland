package com.nearme.webview.plugin;

public class PluginConfig {
	
	public final String mLauncherClassName;
	
	public final String mPluginName;
	
	public final String mPackageName;
	
	public static String sTokenKey;

	public static String sAppKey;
	
	public static String sProductId;
	
	private static final String DEFAULT_PLUGIN_FILE_NAME = "NMWebViewPluginClient.apk";
	
	private static final String DEFAULT_LAUNCHER_CLASS_NAME = "com.nearme.plugin.webview.client.WebActivity";
	
	private static final String DEFAULT_PACKAGE_NAME = "com.nearme.plugin.webview.client";
	
	public PluginConfig(Builder builder) {
		this.mLauncherClassName = builder.mLauncherClassName;
		this.mPluginName = builder.mPluginName;
		this.mPackageName = builder.mPackageName;
		sAppKey = builder.mAppKey;
		sProductId = builder.mProductId;
		sTokenKey = builder.mTokenKey;
	}
	
	public static class Builder {
		private String mLauncherClassName;
		private String mPluginName;
		private String mPackageName;
		private String mAppKey = "";
		private String mProductId = "";
		private String mTokenKey = "";
		
		
		public Builder addLauncherClassName(String name) {
			this.mLauncherClassName = name;
			return this;
		}
		
		public Builder addPluginName(String name) {
			this.mPluginName = name;
			return this;
		}
		
		public Builder addPackageName(String name) {
			this.mPackageName = name;
			return this;
		}
		
		public Builder addTokenKey(String token) {
			this.mTokenKey = token;
			return this;
		}
		
		public Builder addAppKey(String  appKey) {
			this.mAppKey = appKey;
			return this;
		}
		
		public Builder addProductId(String productId) {
			this.mProductId = productId;
			return this;
		}
		
		public PluginConfig build() {
			initEmptyConfig();
			return new PluginConfig(this);
		}
		
		private void initEmptyConfig() {
			
			if (mLauncherClassName == null) {
				mLauncherClassName = DEFAULT_LAUNCHER_CLASS_NAME;
			}
			
			if (mPluginName == null) {
				mPluginName = DEFAULT_PLUGIN_FILE_NAME;
			}
			
			if (mPackageName == null) {
				mPackageName = DEFAULT_PACKAGE_NAME;
			}
		}
		
	}
}
