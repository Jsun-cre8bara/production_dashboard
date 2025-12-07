import React from 'react';
import Header from '../components/layout/Header';
import Sidebar from '../components/layout/Sidebar';
import KpiCard from '../components/common/KpiCard';
import Panel from '../components/common/Panel';

const kpis = [
  '오늘 예매 수',
  '총 매출',
  '진행 공연 수',
  '예매율',
  '공석률',
  '신규 유입',
];

const Dashboard: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 px-8 py-8">
          <div className="max-w-[1440px] mx-auto space-y-8">
            <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {kpis.map((title) => (
                <KpiCard key={title} title={title} />
              ))}
            </section>

            <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <Panel
                  title="공연별 예매 추이"
                  action={
                    <div className="flex items-center space-x-3">
                      <select className="px-3 py-2 text-sm border border-gray-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500">
                        <option>전체 공연</option>
                        <option>뮤지컬</option>
                        <option>콘서트</option>
                      </select>
                      <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden">
                        {['일', '주', '월'].map((label) => (
                          <button
                            key={label}
                            className="px-3 py-2 text-sm bg-white hover:bg-gray-50 focus:outline-none"
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </div>
                  }
                >
                  <div className="h-64 rounded-lg bg-gradient-to-br from-indigo-50 to-white border border-dashed border-indigo-100 flex items-center justify-center text-sm text-indigo-400">
                    라인 차트 영역
                  </div>
                </Panel>
              </div>

              <div className="lg:col-span-1 space-y-6">
                <Panel title="AI 관객 인사이트">
                  <div className="grid grid-cols-1 gap-3 text-sm text-gray-700">
                    {[['연령대', '20대 후반 · 30대 초반 집중'], ['성별', '여성 62% · 남성 38%'], ['지역', '수도권 78% · 지방 22%'], ['추천 관객 유형', '연인 · 친구 · 문화소비 활발 그룹']].map(([label, detail]) => (
                      <div key={label} className="p-3 rounded-lg border border-gray-100 bg-gray-50">
                        <div className="text-xs text-gray-500 mb-1">{label}</div>
                        <div className="font-medium text-gray-800">{detail}</div>
                      </div>
                    ))}
                  </div>
                </Panel>
              </div>
            </section>

            <section>
              <Panel title="좌석·티켓 운영 현황" action={<button className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700">상세 좌석 관리</button>}>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div className="h-64 rounded-lg bg-gray-100 border border-dashed border-gray-200 flex items-center justify-center text-sm text-gray-500">
                    좌석맵 프리뷰
                  </div>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      {[['총 좌석', '1,250'], ['판매', '980'], ['잔여', '270'], ['구역 수', '5']].map(([label, value]) => (
                        <div key={label} className="p-3 rounded-lg bg-gray-50 border border-gray-100 flex flex-col">
                          <span className="text-xs text-gray-500">{label}</span>
                          <span className="text-base font-semibold text-gray-900">{value}</span>
                        </div>
                      ))}
                    </div>
                    <div className="space-y-2 text-sm text-gray-700">
                      {['A 구역 - 92%', 'B 구역 - 88%', 'C 구역 - 75%', 'D 구역 - 67%', 'E 구역 - 54%'].map((zone) => (
                        <div key={zone} className="p-3 rounded-lg border border-gray-100 bg-white flex justify-between">
                          <span className="text-gray-600">{zone.split(' - ')[0]}</span>
                          <span className="font-semibold text-indigo-600">{zone.split(' - ')[1]}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Panel>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
