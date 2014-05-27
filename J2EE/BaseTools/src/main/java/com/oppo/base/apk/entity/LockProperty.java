package com.oppo.base.apk.entity;

/**
 * 
 * ClassName:LockProperty
 * Function:锁屏相关属性
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-18  下午03:08:03
 * @see
 */
public class LockProperty {
	private String name;   //锁屏名称
	private String author; //锁屏作者
	private String version;//锁屏版本
	private String resolution; //锁屏支持的分辨率
	private String description; //锁屏描述
	private String thumbnail;   //该锁屏展示给用户的小缩略图
	private String thumbLeft;  //第一张预览图片
	private String thumbRight;  //第二章预览图片
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public String getAuthor() {
		return author;
	}
	public void setAuthor(String author) {
		this.author = author;
	}
	public String getVersion() {
		return version;
	}
	public void setVersion(String version) {
		this.version = version;
	}
	public String getResolution() {
		return resolution;
	}
	public void setResolution(String resolution) {
		this.resolution = resolution;
	}
	public String getDescription() {
		return description;
	}
	public void setDescription(String description) {
		this.description = description;
	}
	public String getThumbnail() {
		return thumbnail;
	}
	public void setThumbnail(String thumbnail) {
		this.thumbnail = thumbnail;
	}
	public String getThumbLeft() {
		return thumbLeft;
	}
	public void setThumbLeft(String thumbLeft) {
		this.thumbLeft = thumbLeft;
	}
	public String getThumbRight() {
		return thumbRight;
	}
	public void setThumbRight(String thumbRight) {
		this.thumbRight = thumbRight;
	}
}
