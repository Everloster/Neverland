package com.nearme.freeupgrade.lib;

/**
 * OPPO Widget 插件描述文件对应资源ID
 */
public class AppWidgetProviderInfo {

	/**
	 * 最小宽度
	 */
	private int mMinWidth;
	/**
	 * 最小高度
	 */
	private int mMinHeight;
	/**
	 * 更新时间
	 */
	private int mUpdatePeriodMillis;
	/**
	 * 预览图
	 */
	private int mPreviewImage;
	/**
	 * 初始布局
	 */
	private int mInitialLayout;
	/**
	 * 伸缩的步进宽度
	 */
	private int mMinResizeWidth;
	/**
	 * 伸缩的步进宽度
	 */
	private int mMinResizeHeight;

	/**
	 * 获取最小宽度资源ID
	 */
	public int getMinWidth() {
		return mMinWidth;
	}

	public void setMinWidth(int minWidth) {
		this.mMinWidth = minWidth;
	}

	/**
	 * 获取最小高度资源ID
	 */
	public int getMinHeight() {
		return mMinHeight;
	}

	public void setMinHeight(int minHeight) {
		this.mMinHeight = minHeight;
	}

	/**
	 * 获取更新时间资源ID，默认为0
	 */
	public int getUpdatePeriodMillis() {
		return mUpdatePeriodMillis;
	}

	public void setUpdatePeriodMillis(int updatePeriodMillis) {
		this.mUpdatePeriodMillis = updatePeriodMillis;
	}

	/**
	 * 获取预览图资源ID
	 */
	public int getPreviewImage() {
		return mPreviewImage;
	}

	public void setPreviewImage(int previewImage) {
		this.mPreviewImage = previewImage;
	}

	/**
	 * 获取初始布局文件资源ID
	 */
	public int getInitialLayout() {
		return mInitialLayout;
	}

	public void setInitialLayout(int initialLayout) {
		this.mInitialLayout = initialLayout;
	}

	/**
	 * 获取伸缩的步进宽度资源ID
	 */
	public int getMinResizeWidth() {
		return mMinResizeWidth;
	}

	public void setmMinResizeWidth(int minResizeWidth) {
		this.mMinResizeWidth = minResizeWidth;
	}

	/**
	 * 获取伸缩的步进高度资源ID
	 */
	public int getMinResizeHeight() {
		return mMinResizeHeight;
	}

	public void setMinResizeHeight(int minResizeHeight) {
		this.mMinResizeHeight = minResizeHeight;
	}

}
