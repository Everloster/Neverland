package org.jabe.neverland.activity;

import java.util.Date;
import java.util.LinkedList;
import java.util.List;

import org.jabe.neverland.R;
import org.jabe.neverland.model.CommonAdapter;
import org.jabe.neverland.model.ListElement;
import org.jabe.neverland.model.SimpleListItem;
import org.jabe.neverland.view.PullRefreshListView;
import org.jabe.neverland.view.PullRefreshListView.OnRefreshListener;

import android.app.Activity;
import android.os.AsyncTask;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.animation.LinearInterpolator;
import android.view.animation.RotateAnimation;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

public class PullRefreshListviewActivity extends Activity {
	
	PullRefreshListView mListview;
	LinearLayout headView ;
	CommonAdapter<String> baseAdapter;
	LinkedList<String> data = new LinkedList<String>();
	
	private TextView lastUpdatedTextView;
	private ProgressBar progressBar;
	private ImageView arrowImageView;
	private TextView tipsTextview;
	private RotateAnimation animation;
	private RotateAnimation reverseAnimation;
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.act_pullrefresh_listview);
		
		animation = new RotateAnimation(0, -180,
				RotateAnimation.RELATIVE_TO_SELF, 0.5f,
				RotateAnimation.RELATIVE_TO_SELF, 0.5f);
		animation.setInterpolator(new LinearInterpolator());
		animation.setDuration(250);
		animation.setFillAfter(true);

		reverseAnimation = new RotateAnimation(-180, 0,
				RotateAnimation.RELATIVE_TO_SELF, 0.5f,
				RotateAnimation.RELATIVE_TO_SELF, 0.5f);
		reverseAnimation.setInterpolator(new LinearInterpolator());
		reverseAnimation.setDuration(200);
		reverseAnimation.setFillAfter(true);
		
		
		mListview = (PullRefreshListView) findViewById(R.id.container);
		
		LayoutInflater inflater = LayoutInflater.from(this);
		headView = (LinearLayout) inflater.inflate(R.layout.act_pullrefresh_headview, null);
		arrowImageView = (ImageView) headView
				.findViewById(R.id.head_arrowImageView);
		arrowImageView.setMinimumWidth(70);
		arrowImageView.setMinimumHeight(50);
		progressBar = (ProgressBar) headView
				.findViewById(R.id.head_progressBar);
		tipsTextview = (TextView) headView.findViewById(R.id.head_tipsTextView);
		lastUpdatedTextView = (TextView) headView
				.findViewById(R.id.head_lastUpdatedTextView);
		lastUpdatedTextView.setText(new Date().toLocaleString());
		
		List<ListElement<String>> allList = new LinkedList<ListElement<String>>();
		SimpleListItem tempListItem = new SimpleListItem(data, this);
		for (int i = 100; i > 0; i --) {
			data.addLast("string" + i);
			allList.add(tempListItem.clone());
		}
		baseAdapter = new CommonAdapter<String>(allList);
		
		mListview.setHeadViewAndAdapter(headView, baseAdapter);
		mListview.setonRefreshListener(new OnRefreshListener() {
			
			@Override
			public void onStateChange(PullRefreshListView listView, int state, int oldState) {
				
				switch (state) {
				case PullRefreshListView.RELEASE_To_REFRESH:
					
					if (oldState == PullRefreshListView.PULL_To_REFRESH) {
						arrowImageView.clearAnimation();
						arrowImageView.startAnimation(animation);
					}

					tipsTextview.setText("释放加载更多");
					break;
				case PullRefreshListView.PULL_To_REFRESH:
					
					if (oldState == PullRefreshListView.RELEASE_To_REFRESH) {
						arrowImageView.clearAnimation();
						arrowImageView.startAnimation(reverseAnimation);
					}
					
					tipsTextview.setText("下拉加载更多");
					break;

				case PullRefreshListView.REFRESHING:

					progressBar.setVisibility(View.VISIBLE);
					arrowImageView.clearAnimation();
					arrowImageView.setVisibility(View.GONE);
					tipsTextview.setText("加载中...");

					break;
				case PullRefreshListView.DONE:

					progressBar.setVisibility(View.GONE);
					arrowImageView.clearAnimation();
					arrowImageView.setImageResource(R.drawable.arrow);
					arrowImageView.setVisibility(View.VISIBLE);
					tipsTextview.setText("DONE");
					break;
				}
				headView.invalidate();
			}
			
			@Override
			public void onRefresh() {
				new AsyncTask<Object, Object, Object>() {

					@SuppressWarnings("static-access")
					@Override
					protected Object doInBackground(Object... params) {
						try {
							Thread.currentThread().sleep(1000);
						} catch (Exception e) {
							
						}
						final int size = baseAdapter.getCount() + 1;
						baseAdapter.addFirst("string" + size);
						return null;
					}
					
					protected void onPostExecute(Object result) {
						mListview.goneHeadView();
						lastUpdatedTextView.setText(new Date().toLocaleString());
						baseAdapter.notifyDataSetChanged();
					};
					
				}.execute();
			}
		});
	}
	
}
