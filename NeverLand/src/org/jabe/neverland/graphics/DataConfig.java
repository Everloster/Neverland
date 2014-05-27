package org.jabe.neverland.graphics;

public class DataConfig {

	/**
	 * 缩放比例
	 */
	public final static float TOUCH_SCALE_FACTOR = 180.0f/320;
	
	/**
	 * 打开灯光效果标识
	 * false:关灯
	 * true:开灯
	 */
	public static boolean openLightFlag = false;
	
	/**
	 * 上一次触控位置的Y坐标
	 */
	public static float mPreviousY;
	
	/**
	 * 上一次触控位置的X坐标
	 */
	public static float mPreviousX;
	
	/**
	 * 在X轴上的旋转角度
	 */
	public static float mAngleX;
	
	/**
	 * 在Y轴上的旋转角度
	 */
	public static float mAngleY;
	
	/**
	 * 放缩比例/移动比例
	 */
	public static float change = 0.2f;
	
	/**
	 * 缩放矩阵x轴参数
	 */
	public static float scaleX = 1.51f;
	/**
	 * 缩放矩阵y轴参数
	 */
	public static float scaleY = 1.51f;
	/**
	 * 缩放矩阵z轴参数
	 */
	public static float scaleZ = 1.51f;
	
	/**
	 * 移动矩阵x轴参数
	 */
	public static float moveX = 0.0f;
	
	/**
	 * 移动矩阵y轴参数
	 */
	public static float moveY = 0.0f;
	
}
