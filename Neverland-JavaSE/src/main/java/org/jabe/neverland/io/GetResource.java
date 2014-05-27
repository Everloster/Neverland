package org.jabe.neverland.io;


public class GetResource {
	
	public static String PLUGIN_PACKAGE_NAME = "";
	
	public static int getStringResource (String name){
		try {
			final Class<?> clazz = Class.forName(PLUGIN_PACKAGE_NAME+".R$string");
			final Object obj = clazz.newInstance();
			return clazz.getDeclaredField(name).getInt(obj);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		} 
	}
	
	public static int getLayoutResource (String name){
		try {
			final Class<?> clazz = Class.forName(PLUGIN_PACKAGE_NAME+".R$layout");
			final Object obj = clazz.newInstance();
			return clazz.getDeclaredField(name).getInt(obj);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		} 
	}
	
	public static int getDrawableResource (String name){
		try {
			final Class<?> clazz = Class.forName(PLUGIN_PACKAGE_NAME+".R$drawable");
			final Object obj = clazz.newInstance();
			return clazz.getDeclaredField(name).getInt(obj);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		} 
	}
	
	public static int getIdResource (String name){
		try {
			final Class<?> clazz = Class.forName(PLUGIN_PACKAGE_NAME+".R$id");
			final Object obj = clazz.newInstance();
			return clazz.getDeclaredField(name).getInt(obj);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		} 
	}
	
	public static int getDimenResource (String name){
		try {
			final Class<?> clazz = Class.forName(PLUGIN_PACKAGE_NAME+".R$dimen");
			final Object obj = clazz.newInstance();
			return clazz.getDeclaredField(name).getInt(obj);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		} 
	}
	
	public static int getAnimResource (String name){
		try {
			final Class<?> clazz = Class.forName(PLUGIN_PACKAGE_NAME+".R$anim");
			final Object obj = clazz.newInstance();
			return clazz.getDeclaredField(name).getInt(obj);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		} 
	}
	
	public static int getStyleResource (String name){
		try {
			final Class<?> clazz = Class.forName(PLUGIN_PACKAGE_NAME+".R$style");
			final Object obj = clazz.newInstance();
			return clazz.getDeclaredField(name).getInt(obj);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		} 
	}
	
	public static int getStyleableResource(String name) {
		try {
			final Class<?> clazz = Class.forName(PLUGIN_PACKAGE_NAME+".R$styleable");
			final Object obj = clazz.newInstance();
			return clazz.getDeclaredField(name).getInt(obj);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		} 
	}
	
	public static int[] getStyleableArrayResource(String name) {
		try {
			final Class<?> clazz = Class.forName(PLUGIN_PACKAGE_NAME+".R$styleable");
			final Object obj = clazz.newInstance();
			return (int[]) clazz.getDeclaredField(name).get(obj);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		} 
	}
}
