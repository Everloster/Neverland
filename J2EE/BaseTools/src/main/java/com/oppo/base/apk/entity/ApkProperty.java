package com.oppo.base.apk.entity;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * ClassName:ApkProperty
 * Function:APK相关属性
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-7-18  下午03:08:03
 * @see
 */
public class ApkProperty {
	private int minSdkVersion;
	private String packageName;
	private int versionCode;
	private String versionName;
	private String label;
	private String iconAddress;
	private List<String> permissionList;
	private boolean fromGamecenter;

	public ApkProperty() {
		this.permissionList = new ArrayList<String>(10);
	}

	/**
	 * 兼容的最低的sdk版本
	 * @return
	 */
	public int getMinSdkVersion() {
		return minSdkVersion;
	}

	public void setMinSdkVersion(int minSdkVersion) {
		this.minSdkVersion = minSdkVersion;
	}

	/**
	 * 获取包名
	 * @return
	 */
	public String getPackageName() {
		return packageName;
	}

	public void setPackageName(String packageName) {
		this.packageName = packageName;
	}

	public int getVersionCode() {
		return versionCode;
	}

	public void setVersionCode(int versionCode) {
		this.versionCode = versionCode;
	}

	public String getVersionName() {
		return versionName;
	}

	public void setVersionName(String versionName) {
		this.versionName = versionName;
	}

	/**
	 * 获取label
	 * @return  the label
	 * @since   Ver 1.0
	 */
	public String getLabel() {
		return label;
	}

	/**
	 * 设置label
	 * @param   label
	 * @since   Ver 1.0
	 */
	public void setLabel(String label) {
		this.label = label;
	}

	public String getIconAddress() {
		return iconAddress;
	}

	public void setIconAddress(String iconAddress) {
		this.iconAddress = iconAddress;
	}

	/**
	 * 获取permissionList
	 * @return  the permissionList
	 * @since   Ver 1.0
	 */
	public List<String> getPermissionList() {
		return permissionList;
	}

	/**
	 * 设置permissionList
	 * @param   permissionList
	 * @since   Ver 1.0
	 */
	public void setPermissionList(List<String> permissionList) {
		this.permissionList = permissionList;
	}

	/**
	 * 获取是否来自游戏中心
	 * @return  the fromGamecenter
	 * @since   Ver 1.0
	 */
	public boolean isFromGamecenter() {
		return fromGamecenter;
	}

	/**
	 * 设置是否来自游戏中心
	 * @param   fromGamecenter
	 * @since   Ver 1.0
	 */
	public void setFromGamecenter(boolean fromGamecenter) {
		this.fromGamecenter = fromGamecenter;
	}

	public void addPermission(String permission) {
		if(null == this.permissionList) {
			this.permissionList = new ArrayList<String>(10);
		}

		this.permissionList.add(permission);
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("package name:").append(this.packageName);
		sb.append(",version code:").append(this.versionCode);
		sb.append(",version name:").append(this.versionName);
		sb.append(",min sdk version:").append(this.minSdkVersion);
		sb.append(",label:").append(this.label);
		sb.append(",icon path:").append(this.iconAddress);
		sb.append(",permission:");
		int permListLen = (null == this.permissionList) ? 0 : this.permissionList.size();
		if(permListLen > 0) {
			for(int i = 0; i < permListLen; i++) {
				sb.append(permissionList.get(i)).append('|');
			}
		}
		sb.append(",from gamecenter:").append(fromGamecenter);

		return sb.toString();
	}
}
