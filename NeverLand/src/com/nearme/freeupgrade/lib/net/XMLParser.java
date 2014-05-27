/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-8-19
 */
package com.nearme.freeupgrade.lib.net;

import java.io.StringReader;

import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserFactory;

import android.text.TextUtils;

import com.nearme.freeupgrade.lib.pinterface.PluginItem;
import com.nearme.freeupgrade.lib.pinterface.Plugins;

/**
 * 
 * @Author liu jing qiang
 * @Since 2013-8-19
 */
public class XMLParser {
	/**
	 * Get attribute value
	 * 
	 * @param parser
	 *            XML pull parser
	 * @param mName
	 *            attribute mName
	 * @return attribute value, if attribute doesn't exist, return empty string
	 */
	private static String getAttributeValue(XmlPullParser parser, String name) {
		String value = parser.getAttributeValue(XmlPullParser.NO_NAMESPACE, name);
		return value == null ? "" : value.trim();
	}

	/**
	 * Skip an unknown tag. Current event must be START_TAG before calling this
	 * method. After returned, current event is the corresponding END_TAG
	 * 
	 * @param parser
	 *            XmlPullParser
	 * @throws Exception
	 */
	private static void skipUnknownTag(XmlPullParser parser) throws Exception {
		String tag = parser.getName();
		while (parser.next() > 0) {
			if (parser.getEventType() == XmlPullParser.END_TAG && parser.getName().equals(tag))
				break;
		}
	}

	private static int getIntAttribute(XmlPullParser parser, String name) {
		try {
			return Integer.parseInt(getAttributeValue(parser, name).trim());
		} catch (Exception e) {
			return 0;
		}
	}

	private static long getLongAttribute(XmlPullParser parser, String name) {
		try {
			return Long.parseLong(getAttributeValue(parser, name).trim());
		} catch (Exception e) {
			return 0;
		}
	}

	public static Plugins parsePlugins(String body) throws Exception {
		if (TextUtils.isEmpty(body))
			return null;
		// create XML pull parser
		XmlPullParser parser = XmlPullParserFactory.newInstance().newPullParser();
		parser.setInput(new StringReader(body));
		// create result bean
		Plugins result = null;
		parser.nextTag();
		parser.require(XmlPullParser.START_TAG, XmlPullParser.NO_NAMESPACE, "response");

		while (parser.nextTag() == XmlPullParser.START_TAG) {
			String nodeName1 = parser.getName();
			parser.require(XmlPullParser.START_TAG, XmlPullParser.NO_NAMESPACE, nodeName1);

			if ("plugins".equals(nodeName1)) {
				result = new Plugins();
				while (parser.nextTag() == XmlPullParser.START_TAG) {
					String nodeName2 = parser.getName();
					parser.require(XmlPullParser.START_TAG, XmlPullParser.NO_NAMESPACE, nodeName2);
					if ("plugin".equals(nodeName2)) {
						PluginItem item = new PluginItem();
						item.upgradeType = getIntAttribute(parser, "upgrade_type");
						item.pid = getAttributeValue(parser, "id");
						item.name = getAttributeValue(parser, "name");
						item.pkgName = getAttributeValue(parser, "package_name");
						item.iconUrl = getAttributeValue(parser, "icon_url");
						item.fileUrl = getAttributeValue(parser, "file_url");
						item.fileSize = getLongAttribute(parser, "file_size");
						item.fileMd5 = getAttributeValue(parser, "file_md5");
						item.versionCode = getIntAttribute(parser, "version_code");
						item.versionName = getAttributeValue(parser, "version_name");
						item.desc = getAttributeValue(parser, "desc");
						item.upgradeComment = getAttributeValue(parser, "upgrade_comment");

						result.pluginList.add(item);
						parser.nextTag();
					} else
						skipUnknownTag(parser);

					// tag should end
					parser.require(XmlPullParser.END_TAG, XmlPullParser.NO_NAMESPACE, nodeName2);
				}
			} else
				skipUnknownTag(parser);

			// tag should end
			parser.require(XmlPullParser.END_TAG, XmlPullParser.NO_NAMESPACE, nodeName1);
		}

		// results should end
		parser.require(XmlPullParser.END_TAG, XmlPullParser.NO_NAMESPACE, "response");

		return result;
	}

}
