/**
 * ThemeInfo.java
 * com.oppo.base.theme
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-28 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.theme;

import com.oppo.base.common.NumericUtil;
import com.oppo.base.common.OConstants;

/**
 * ClassName:ThemeInfo
 * Function: 主题信息
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-28  下午04:10:39
 */
public class ThemeInfo {
	private String model;		//机型
    private String versionName;	//版本
    private String author;		//主题作者
    private String description;	//主题简介
    private String name;		//主题名
    private int width;			//
    private int height;			//
    private String editorVersion;	//编辑器版本
    private String resolution;		//分辨率
    
    //新主题新加属性
    private String packageName;   //包名
    
    /**
	 * 获取机型
	 * @return  the model
	 * @since   Ver 1.0
	 */
	public String getModel() {
		return model;
	}
	/**
	 * 设置机型
	 * @param   model    
	 * @since   Ver 1.0
	 */
	public void setModel(String model) {
		this.model = model;
	}
	/**
	 * 获取版本
	 * @return  the versionName
	 * @since   Ver 1.0
	 */
	public String getVersionName() {
		return versionName;
	}
	/**
	 * 设置版本
	 * @param   versionName    
	 * @since   Ver 1.0
	 */
	public void setVersionName(String versionName) {
		this.versionName = versionName;
	}
	/**
	 * 获取主题作者
	 * @return  the author
	 * @since   Ver 1.0
	 */
	public String getAuthor() {
		return author;
	}
	/**
	 * 设置主题作者
	 * @param   author    
	 * @since   Ver 1.0
	 */
	public void setAuthor(String author) {
		this.author = author;
	}
	/**
	 * 获取主题简介
	 * @return  the description
	 * @since   Ver 1.0
	 */
	public String getDescription() {
		return description;
	}
	/**
	 * 设置主题简介
	 * @param   description    
	 * @since   Ver 1.0
	 */
	public void setDescription(String description) {
		this.description = description;
	}
	/**
	 * 获取主题名
	 * @return  the name
	 * @since   Ver 1.0
	 */
	public String getName() {
		return name;
	}
	/**
	 * 设置主题名
	 * @param   name    
	 * @since   Ver 1.0
	 */
	public void setName(String name) {
		this.name = name;
	}
	/**
	 * 获取width
	 * @return  the width
	 * @since   Ver 1.0
	 */
	public int getWidth() {
		return width;
	}
	/**
	 * 设置width
	 * @param   width    
	 * @since   Ver 1.0
	 */
	public void setWidth(int width) {
		this.width = width;
	}
	/**
	 * 获取height
	 * @return  the height
	 * @since   Ver 1.0
	 */
	public int getHeight() {
		return height;
	}
	/**
	 * 设置height
	 * @param   height    
	 * @since   Ver 1.0
	 */
	public void setHeight(int height) {
		this.height = height;
	}
	/**
	 * 获取editorVersion
	 * @return  the editorVersion
	 * @since   Ver 1.0
	 */
	public String getEditorVersion() {
		return editorVersion;
	}
	/**
	 * 设置editorVersion
	 * @param   editorVersion    
	 * @since   Ver 1.0
	 */
	public void setEditorVersion(String editorVersion) {
		this.editorVersion = editorVersion;
	}
	/**
	 * 获取resolution
	 * @return  the resolution
	 * @since   Ver 1.0
	 */
	public String getResolution() {
		return resolution;
	}
	/**
	 * 设置resolution
	 * @param   resolution    
	 * @since   Ver 1.0
	 */
	public void setResolution(String resolution) {
		this.resolution = resolution;
	}
	
	/**
	 * 获取os
	 * @param 
	 * @return
	 */
	public int getOs() {
		if(null == this.editorVersion) {
			return 0;
		}
		
		String osString = this.editorVersion;
		int index = osString.indexOf(".");
		if(index > 0) {
			osString = osString.substring(0, index);
		}
		
		return NumericUtil.parseInt(osString, 0);
	}
	
	/**
	 * 获取机型
	 * @param 
	 * @return
	 */
	public String getScreen() {
		if(null == resolution) {
			return OConstants.EMPTY_STRING;
		}
		
		return resolution.replace('x', '#');
	}
	
	/**
	 * 获取主题信息的String描述
	 * @param 
	 * @return
	 */
	public String getThemeDescription() {
		StringBuilder sb = new StringBuilder();
		sb.append("model:").append(model);
		sb.append(",versionName:").append(versionName);
		sb.append(",width:").append(width);
		sb.append(",height:").append(height);
		sb.append(",name:").append(name);
		sb.append(",author:").append(author);
		sb.append(",description:").append(description);	
		sb.append(",editorVersion:").append(editorVersion);
		sb.append(",resolution:").append(resolution);
		sb.append(",os:").append(getOs());
		sb.append(",screen:").append(getScreen());
		
		return sb.toString();
	}
	public String getPackageName() {
		return packageName;
	}
	public void setPackageName(String packageName) {
		this.packageName = packageName;
	}
}

