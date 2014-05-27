/**
 * ApkResourceProperty.java
 * com.oppo.base.apk
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-18 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.apk.entity;

import com.oppo.base.common.StringUtil;

/**
 * ClassName:ApkResourceProperty
 * Function: apk id资源
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-7-18  下午03:08:26
 */
public class ApkResourceProperty {
	private static final String SPLIT_CHAR = "/";
	private String folder;
	private String resource;
	private String address;

	public ApkResourceProperty(){
	}

	public ApkResourceProperty(String folder, String resource, String address) {
		this.folder = folder;
		this.resource = resource;
		this.address = address;
	}

	public boolean isFit(String fileName) {
		if(StringUtil.isNullOrEmpty(fileName)) {
			return false;
		}

		int splitIndex = fileName.lastIndexOf(SPLIT_CHAR);
		if(splitIndex < 1) {
			return false;
		}

		String fFolder = fileName.substring(0, splitIndex);
		if(!fFolder.equals(folder)) {
			return false;
		}

		String fRes = fileName.substring(splitIndex + 1);
		if(!fRes.equals(resource)) {
			return false;
		} else {
			return true;
		}
	}

	/**
	 * 获取资源目录
	 * @return  the folder
	 * @since   Ver 1.0
	 */
	public String getFolder() {
		return folder;
	}

	/**
	 * 设置资源目录
	 * @param   folder
	 * @since   Ver 1.0
	 */
	public void setFolder(String folder) {
		this.folder = folder;
	}

	/**
	 * 获取资源名(无后缀)
	 * @return  the resource
	 * @since   Ver 1.0
	 */
	public String getResource() {
		return resource;
	}

	/**
	 * 设置资源名(无后缀)
	 * @param   resource
	 * @since   Ver 1.0
	 */
	public void setResource(String resource) {
		this.resource = resource;
	}
	/**
	 * 获取资源地质
	 * @return  the address
	 * @since   Ver 1.0
	 */
	public String getAddress() {
		return address;
	}
	/**
	 * 设置资源地质
	 * @param   address
	 * @since   Ver 1.0
	 */
	public void setAddress(String address) {
		this.address = address;
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("folder:").append(this.folder);
		sb.append(",resource:").append(this.resource);
		sb.append(",address:").append(this.address);

		return sb.toString();
	}
}

