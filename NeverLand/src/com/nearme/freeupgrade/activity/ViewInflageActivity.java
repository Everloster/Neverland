package com.nearme.freeupgrade.activity;

import java.io.File;
import java.io.IOException;

import org.xmlpull.v1.XmlPullParserException;

import android.app.Activity;
import android.content.pm.PackageManager.NameNotFoundException;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;

import com.nearme.freeupgrade.lib.pinterface.PluginInterface;
import com.nearme.freeupgrade.lib.util.Constants;

public class ViewInflageActivity extends Activity {

	private final static String TAG = ViewInflageActivity.class.getSimpleName();

	private final static String INITIAL_LAYOUT_NAME = "search_widget_info";

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		Drawable drawable = new ColorDrawable(Color.BLACK);
		this.getWindow().setBackgroundDrawable(drawable);
		setContentView(getWidgetView());
	}

	private View getWidgetView() {
		String apkPath = Environment.getExternalStorageDirectory() + File.separator
				+ Constants.ROOT_DIR_NAME + File.separator + "OppoSearchBox.apk";
		try {
			return PluginInterface.getWidgetViewByXml(this, apkPath, INITIAL_LAYOUT_NAME);
		} catch (NameNotFoundException e) {
			e.printStackTrace();
		} catch (XmlPullParserException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
}
