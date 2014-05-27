package com.mobeta.android.demodslv;

import java.util.ArrayList;
import java.util.Arrays;

import org.jabe.neverland.R;

import android.app.ListActivity;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;

//import com.mobeta.android.demodslv.R;

public class Launcher extends ListActivity {

	// private ArrayAdapter<ActivityInfo> adapter;
	private MyAdapter adapter;

	private ArrayList<ActivityInfo> mActivities = new ArrayList<ActivityInfo>();

	private String[] mActTitles;
	private String[] mActDescs;

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.launcher);

		try {
			PackageInfo pi = getPackageManager().getPackageInfo(
					"org.jabe.neverland",
					PackageManager.GET_ACTIVITIES);

			ArrayList<ActivityInfo> temp = new ArrayList<ActivityInfo>(
					Arrays.asList(pi.activities));
			String ourName = getClass().getName();
			for (int i = 0; i < temp.size(); ++i) {
				if (!ourName.equals(temp.get(i).name) && temp.get(i).name.contains("com.mobeta.android.demodslv")) {
					mActivities.add(temp.get(i));
				}
			}
		} catch (PackageManager.NameNotFoundException e) {
			// Do nothing. Adapter will be empty.
		}

		mActTitles = getResources().getStringArray(R.array.activity_titles);
		mActDescs = getResources().getStringArray(R.array.activity_descs);

		// adapter = new ArrayAdapter<ActivityInfo>(this,
		// R.layout.launcher_item, R.id.text, mActivities);
		adapter = new MyAdapter();

		setListAdapter(adapter);
	}

	@Override
	protected void onListItemClick(ListView l, View v, int position, long id) {
		Intent intent = new Intent();
		intent.setClassName(this, mActivities.get(position).name);
		startActivity(intent);
	}

	private class MyAdapter extends ArrayAdapter<ActivityInfo> {
		MyAdapter() {
			super(Launcher.this, R.layout.launcher_item, R.id.activity_title,
					mActivities);
		}
			
		@Override
		public int getCount() {
			return mActTitles.length;
		}

		/* (non-Javadoc)
		 * @see android.widget.ArrayAdapter#getItem(int)
		 */
		@Override
		public ActivityInfo getItem(int position) {
			try {
				return super.getItem(position);
			} catch (Exception e) {
			}
			return null;
		}

		@Override
		public View getView(int position, View convertView, ViewGroup parent) {
			View v = super.getView(position, convertView, parent);

			TextView title = (TextView) v.findViewById(R.id.activity_title);
			TextView desc = (TextView) v.findViewById(R.id.activity_desc);

			title.setText(mActTitles[position]);
			desc.setText(mActDescs[position]);
			return v;
		}

	}

}
